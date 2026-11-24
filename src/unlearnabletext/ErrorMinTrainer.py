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
                 cos_similarity_threshold, 
                 update_rate, train_ids, 
                 tokenizer: AutoTokenizer, 
                 compute_metrics,
                 preprocess_logits_for_metrics,
                 patience=0.00
                ):
        
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.cos_similarity_threshold = cos_similarity_threshold
        self.update_rate = update_rate
        self.noise = {}
        self.token_to_id = {tuple(sample["input_ids"]): arc_id for arc_id, sample in zip(train_ids, train_dataset)}
        self.tokenizer = tokenizer
        self.modified_examples = {}
        self.patience=patience
        self.delta_loss = np.array([])
        
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
            
            norm_input_grads = F.normalize(inputs_embeds[index], dim=1)
            norm_embedding_matrix = F.normalize(embedding_matrix, dim=1)
            
            cos_sim = norm_input_grads @ norm_embedding_matrix.T
            valid_mask = cos_sim >= self.cos_similarity_threshold
            
            replacement_matrix = input_grads[index] @ embedding_matrix.T
            original_scores = (input_grads[index] * inputs_embeds[index]).sum(dim=-1, keepdim=True)
            replacement_matrix = replacement_matrix - original_scores
            replacement_matrix[~valid_mask] = float("inf")
            
            start = starts[index]
            
            # Only include question in replacement matrix
            replacement_matrix = replacement_matrix[start+1:start + question_length]
            # Only check first 10000 for computational efficiency
            flat_scores = replacement_matrix.flatten()
            top_values, sorted_indices = torch.topk(
                flat_scores,
                k=5000,
                largest=False,
            )
            # Dividing by the number of columns gives the row, which corresponds to the sequence position
            positions = sorted_indices // replacement_matrix.shape[1]
            # taking mod gives the column, which corresponds to the replacement embedding candidate
            replacements = sorted_indices % replacement_matrix.shape[1]
            
            # Iterate through the sorted list and look for candidates that meet similarity requirement
            did_find = False
            for position, replacement in zip(positions, replacements):
                # Accounts for start token removal
                absolute_position = position.item() + start + 1
                replacement_id = replacement.item()
                
                # Here we want  to make sure the modification survives retokenization
                modified_ids = inputs["input_ids"][index].clone()
                modified_ids[absolute_position] = replacement_id

                attention_mask = inputs["attention_mask"][index].bool()
                expected_ids = modified_ids[attention_mask].cpu()

                # Decode the modified sequence
                modified_text = self.tokenizer.decode(expected_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)

                retokenized_ids = self.tokenizer(modified_text,add_special_tokens=False)["input_ids"]

                special_ids = set(self.tokenizer.all_special_ids)
                expected_ids = torch.tensor([token_id.item(),for token_id in expected_ids,if token_id.item() not in special_ids])
                survived = torch.equal(torch.tensor(retokenized_ids),expected_ids)
                
                cos_sim = F.cosine_similarity(inputs_embeds[index, absolute_position], embedding_matrix[replacement_id], dim=0).item()
                # Only consider modifications to the question, not choices.
                if survived and inputs["input_ids"][index][absolute_position] not in self.tokenizer.all_special_ids and replacement_matrix[position,replacement] < float("inf") and replacement_id not in self.tokenizer.all_special_ids:
                    
                    did_find = True
                    self.noise[arc_ids[index]] = (absolute_position - starts[index], replacement.item())
                    self.delta_loss = np.append(self.delta_loss, replacement_matrix[position, replacement].item())
                    break
                    
            if not did_find:
                print("No replacement found")
                self.noise.pop(arc_ids[index],None) 
    
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
            if not self.control.should_training_stop:
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
