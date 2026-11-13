from transformers import Trainer
import torch.nn.functional as F
import torch
import numpy as np




# Class for generating error-minimizing perturbations
class ErrorMinTrainer(Trainer):
    replacements = []
    eval_loss = None
    
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, similarity_threshold, update_rate):
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
        self.similarity_threshold = similarity_threshold
        self.update_rate = update_rate
    
    def optimize_noise(self, model, inputs):
        # Get embedding matrix
        embedding_layer = model.get_input_embeddings()
        embedding_matrix = embedding_layer.weight

        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
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
        sample_ids = inputs.pop("sample_id")
        for index, (sample, sample_id) in enumerate(zip(input_grads, sample_ids)):
            
            # First order approximation for token substitutions that minimize error.
            replacement_matrix = sample @ embedding_matrix.T
            flat_scores = replacement_matrix.flatten()
            sorted_indices = torch.argsort(flat_scores)
            
            # Dividing by the number of columns gives the row, which corresponds to the sequence position
            positions = sorted_indices // replacement_matrix.shape[1]
            # taking mod gives the column, which corresponds to the replacement embedding candidate
            replacements = sorted_indices % replacement_matrix.shape[1]
            
            # Iterate through the sorted list and look for candidates that meet similarity requirement
            for position, replacement in zip(positions, replacements):
                cos_sim = F.cosine_similarity(inputs_embeds[position], embedding_matrix[replacement], dim=0).item()
                if cos_sim >= self.similarity_threshold:
                    self.replacements.append((sample_id,position,replacement))
                    inputs["input_ids"][index, self.target_index] = replacement
                    break
    
    # Runs every batch. If the training step mod update rate is 0
    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.state.global_step % self.update_rate == 0:
            # Test metric
            eval_results = self.evaluate()
            if self.eval_loss is None or eval_results['eval_loss'] > self.eval_loss:
                self.eval_loss = eval_results['eval_loss']
            else:
                self.control.should_training_stop = True
                return
            
            self.optimize_noise(model=model, inputs=inputs)
                    
        # Complete normal training, updating grad accumulation
        return super().training_step(model, inputs, num_items_in_batch)
