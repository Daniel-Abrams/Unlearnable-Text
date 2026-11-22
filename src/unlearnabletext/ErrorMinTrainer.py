from transformers import Trainer, AutoTokenizer
import torch.nn.functional as F
import torch
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

# Class for generating error-minimizing perturbations
class ErrorMinTrainer(Trainer):
    def __init__(self, 
                 model, 
                 args, 
                 train_dataset, 
                 eval_dataset, 
                 data_collator, 
                 similarity_threshold, 
                 update_rate, train_ids, 
                 tokenizer: AutoTokenizer, 
                 compute_metrics,
                 preprocess_logits_for_metrics,
                 patience=0.03
                ):
        
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.similarity_threshold = similarity_threshold
        self.update_rate = update_rate
        self.noise = {}
        self.token_to_id = {tuple(sample["input_ids"]): arc_id for arc_id, sample in zip(train_ids, train_dataset)}
        self.tokenizer = tokenizer
        self.modified_examples = {}
        self.similarity_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.patience=patience
        
        self.noise_dataloader = DataLoader(
                                train_dataset,
                                batch_size=8,
                                shuffle=False,
                                collate_fn=data_collator
                            )
        self.eval_token_accuracy = None


    def get_arc_ids(self, inputs):
        arc_ids, starts = [], []
        for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"]):
            start = mask.nonzero()[0].item()
            non_padding = ids[mask.bool()].tolist()
            key = tuple(non_padding)
            arc_ids.append(self.token_to_id[key])
            starts.append(start)
        return arc_ids, starts

    def apply_noise(self, inputs, arc_ids, starts, question_lengths):
        modified_inputs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        s_changed = 0
        for bid, (arc_id, question_length) in enumerate(zip(arc_ids, question_lengths)):
            if arc_id in self.noise:
                start = int(starts[bid])
                question_length = int(question_length)

                pos, token = self.noise[arc_id]
                print(f"Token being replaced: {self.tokenizer.decode(modified_inputs["input_ids"][bid, pos + start])}")
                modified_inputs["input_ids"][bid, pos + start] = token

                self.modified_examples[arc_id] = self.tokenizer.decode(
                    modified_inputs["input_ids"][bid, start:start + question_length],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                ).strip()

                s_changed += 1

        print(f"Samples modified: {s_changed}")

        return modified_inputs
                
                    
    def optimize_noise(self, model, inputs, arc_ids, starts, question_lengths):
        # Get embedding matrix
        embedding_layer = model.get_input_embeddings()
        embedding_matrix = embedding_layer.weight

        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds.requires_grad_(True)
        inputs_embeds.retain_grad()

        # Run a forward pass to compute the initial loss
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )

        # Caluculate the loss
        loss = outputs.loss

        #  Compute gradient of loss w.r.t. inputs_embeds 
        input_grads = torch.autograd.grad(
            outputs=loss,
            inputs=inputs_embeds,
            retain_graph=True,
        )[0]
        
        # Compute and apply the noise for each item in the batch
        for index, (sample,question_length) in enumerate(zip(input_grads,question_lengths)):
            # First order approximation for token substitutions that minimize error.
            replacement_matrix = input_grads[index] @ embedding_matrix.T
            original_scores = (input_grads[index] * inputs_embeds[index]).sum(dim=-1, keepdim=True)
            replacement_matrix = replacement_matrix - original_scores
            start = starts[index]
            
            # Only include question in replacement matrix
            replacement_matrix = replacement_matrix[start+1:start + question_length]
            # Only check first 50000 for computational efficiency
            flat_scores = replacement_matrix.flatten()
            top_values, sorted_indices = torch.topk(
                flat_scores,
                k=50000,
                largest=False
            )
            
            # Dividing by the number of columns gives the row, which corresponds to the sequence position
            positions = sorted_indices // replacement_matrix.shape[1]
            # taking mod gives the column, which corresponds to the replacement embedding candidate
            replacements = sorted_indices % replacement_matrix.shape[1]
            
            # Iterate through the sorted list and look for candidates that meet similarity requirement
            for position, replacement in zip(positions, replacements):
                
                # Accounts for start token removal
                position += 1

                # Only consider modifications to the question, not choices.
                if inputs["input_ids"][index][position] not in self.okenizer.all_special_ids:
                    original = self.tokenizer.decode(t
                        inputs["input_ids"][index, start:start + question_length],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    
                    mod = inputs["input_ids"][index].detach().clone()
                    mod[position] = replacement
                    modified_sentence = self.tokenizer.decode(
                        mod[start:start + question_length],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    # print(f"original: {original}")
                    # print(f"modified: {modified_sentence}")
                    
                    emb1 = self.similarity_model.encode(original)
                    emb2 = self.similarity_model.encode(modified_sentence)
                    score = self.similarity_model.similarity(emb1, emb2) 
                    
                    if score >= self.similarity_threshold:
                        self.noise[arc_ids[index]] = (position.item() - starts[index], replacement.item())
                        #edit our batch outside of the noise optimization step (metches closer to algorithm 1)
                        #inputs["input_ids"][index, position] = replacement]
                        break
    
    def apply_all_noise(self):
        for batch in self.noise_dataloader:
            arc_ids, starts = self.get_arc_ids(batch)
            question_lengths = batch.pop("question_length")
            # Need to move the batch inside the device
            device = next(self.model.parameters()).device
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            # Get IDs and starts for the batch
            self.apply_noise(batch, arc_ids, starts, question_lengths)
    
    # Runs every batch. If the training step mod update rate is 0
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Get info about the current batch
        current_arc_ids, current_starts = self.get_arc_ids(inputs)
        current_question_lengths = inputs.pop("question_length")
        
        # Every M steps we take a metric, decide whether to stop or keep optimizing noise.
        if self.state.global_step % self.update_rate == 0:
            # Test metric
            eval_results = self.evaluate()
            # If the model performed worse than last time, keep going
            if self.eval_token_accuracy is None or eval_results['eval_token_accuracy'] < self.eval_token_accuracy + self.patience:
                if self.eval_token_accuracy:
                    self.eval_token_accuracy = min(eval_results['eval_token_accuracy'], self.eval_token_accuracy)
                else:
                    self.eval_token_accuracy = eval_results['eval_token_accuracy']
            else:
                self.control.should_training_stop = True
                
            print("Evaluation finished")
            print("Starting noise optimization")
            # generate noise for the next batches
            if not self.control.should_training_stop:
                for index, batch in enumerate(self.noise_dataloader):
                    if index >= self.state.global_step and index < self.state.global_step + self.update_rate:
                        # Get IDs and starts for the batch
                        arc_ids, starts = self.get_arc_ids(batch)
                        question_lengths = batch.pop("question_length")
                        # Need to move the batch inside the device
                        device = next(model.parameters()).device
                        batch = {
                            key: value.to(device) if isinstance(value, torch.Tensor) else value
                            for key, value in batch.items()
                        }
                        # Optimize noise
                        self.optimize_noise(model=model, inputs=batch, arc_ids=arc_ids, starts=starts, question_lengths=question_lengths)
            else:
                # Otherwise generate noise for the entire dataset
                for batch in self.noise_dataloader:
                    # Get IDs and starts for the batch
                    arc_ids, starts = self.get_arc_ids(batch)
                    question_lengths = batch.pop("question_length")
                    # Need to move the batch inside the device
                    device = next(model.parameters()).device
                    batch = {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    # Optimize noise
                    self.optimize_noise(model=model, inputs=batch, arc_ids=arc_ids, starts=starts, question_lengths=question_lengths)

        # Apply the current noise (positions, replacements) to the batch
        modified_batch = self.apply_noise(inputs, current_arc_ids, current_starts, current_question_lengths)
        # Complete normal training, updating grad accumulation
        return super().training_step(model, modified_batch, num_items_in_batch)
