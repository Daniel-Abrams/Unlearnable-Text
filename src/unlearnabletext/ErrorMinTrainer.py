from transformers import Trainer
import torch.nn.functional as F
import torch
import numpy as np




# Class for generating error-minimizing perturbations
class ErrorMinTrainer(Trainer):
    #replacements = []
    eval_loss = None
    
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, similarity_threshold, update_rate, train_ids):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
        self.similarity_threshold = similarity_threshold
        self.update_rate = update_rate
        self.noise = {}
        self.token_to_id = {tuple(sample["input_ids"]): arc_id for arc_id, sample in zip(train_ids, train_dataset)}
        

    def get_arc_ids(self, inputs):
        arc_ids, starts = [], []
        for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"]):
            start = mask.nonzero()[0].item()
            non_padding = ids[mask.bool()].tolist()
            key = tuple(non_padding)
            arc_ids.append(self.token_to_id[key])
            starts.append(start)
        return arc_ids, starts

    def apply_noise(self, inputs, arc_ids, starts):
        for bid, arc_id in enumerate(arc_ids):
            if arc_id in self.noise:
                pos, token = self.noise.get(arc_id)
                inputs["input_ids"][bid, pos + starts[bid]] = token

            

    
    def optimize_noise(self, model, inputs, arc_ids, starts):
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
        for index, sample, in enumerate(input_grads):
            # First order approximation for token substitutions that minimize error.
            replacement_matrix = sample @ embedding_matrix.T
            flat_scores = replacement_matrix.flatten()
            top_values, sorted_indices = torch.topk(
                flat_scores,
                k=10000,
                largest=False
            )
            
            # Dividing by the number of columns gives the row, which corresponds to the sequence position
            positions = sorted_indices // replacement_matrix.shape[1]
            # taking mod gives the column, which corresponds to the replacement embedding candidate
            replacements = sorted_indices % replacement_matrix.shape[1]
            
            # Iterate through the sorted list and look for candidates that meet similarity requirement
            for position, replacement in zip(positions, replacements):
                cos_sim = F.cosine_similarity(inputs_embeds[index, position], embedding_matrix[replacement], dim=0).item()
                if cos_sim >= self.similarity_threshold:
                    #self.replacements.append((index,position.item(),replacement.item()))
                    self.noise[arc_ids[index]] = (position.item() - starts[index], replacement.item())
                    #edit our batch outside of the noise optimization step (metches closer to algorithm 1)
                    #inputs["input_ids"][index, position] = replacement
                    break
    
    # Runs every batch. If the training step mod update rate is 0
    def training_step(self, model, inputs, num_items_in_batch=None):
        
        arc_ids, starts = self.get_arc_ids(inputs)

        if self.state.global_step % self.update_rate == 0:
            # Test metric
            eval_results = self.evaluate()
            if self.eval_loss is None or eval_results['eval_loss'] > self.eval_loss:
                self.eval_loss = eval_results['eval_loss']
            else:
                self.control.should_training_stop = True
                
            print("Evaluation finished")
            print("Starting noise optimization")
            self.optimize_noise(model=model, inputs=inputs, arc_ids=arc_ids, starts=starts)
            print("Noise optimization finished")

        self.apply_noise(inputs, arc_ids, starts)
                    
        # Complete normal training, updating grad accumulation
        return super().training_step(model, inputs, num_items_in_batch)
