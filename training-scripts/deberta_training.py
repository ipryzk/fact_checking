import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, matthews_corrcoef
import json
import os
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from collections import Counter
import torch.nn.utils as nn_utils
from torch.amp import GradScaler, Autocast
from pathlib import Path
import time

class LinearDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_initial, lr_final, total_steps, warmup_steps=500, last_epoch=-1):
        self.lr_initial = lr_initial
        self.lr_final = lr_final
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr = self.lr_initial * (step / self.warmup_steps)  # Linear warmup
            print(f"Warmup step {step}: lr = {lr}")
        else:
            lr = self.lr_initial - (self.lr_initial - self.lr_final) * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            print(f"Decay step {step}: lr = {lr}")
        return [lr for _ in self.optimizer.param_groups]

class ClassificationModel:
    def __init__(self, model_name, num_labels=3, checkpoint_dir="./checkpoints", warmup_steps=500, total_steps=5317):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2id = {"corroborate": 0, "contrast": 1, "pass": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.lr = 5e-6 # Adjust LRs depending on learning rate or get a scheduler
        self.lr_initial = 5e-6
        self.lr_final = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # Initialize LinearDecayLR scheduler
        self.scheduler = LinearDecayLR(self.optimizer, warmup_steps=warmup_steps, total_steps=total_steps, lr_initial = self.lr_initial, lr_final = self.lr_final)
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_checkpoint = os.path.join(self.checkpoint_dir, "new.pth")
        self.scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
        print(f"Model and tokenizer loaded. Using device: {self.device}")

    def log_to_json(self, log_data, file_path):
        """
        Append log data to a JSON file. Creates the file if it doesn't exist.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(log_data)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def save_model(self, epoch):
        """
        Save the model and optimizer state to a checkpoint.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(state, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    def load_and_tokenize_data(self, data_path):
        data = torch.load(data_path)
        print(f"Loaded data from {data_path}, total samples: {len(data['input_ids'])}")

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
                print(f"Dataset loaded with {len(self.data['input_ids'])} samples.")

            def __len__(self):
                return len(self.data['input_ids'])

            def __getitem__(self, idx):
                return {
                    'input_ids': self.data['input_ids'][idx],
                    'attention_mask': self.data['attention_mask'][idx],
                    'labels': self.data['labels'][idx].item()
                }

        return CustomDataset(data)
    
    def get_lr(self):
        return self.scheduler.optimizer.param_groups[0]['lr']
    
    def shutdown(self):
        os.system("shutdown.exe /s /f /t 0")

    def get_temperature(self):
        """
        Check the system temperature.
        """
        # Use psutil to get system sensors' temperatures (this was for automatically manaigng  my system)
        sensors = psutil.sensors_temperatures()
        if not sensors:
            return None  # If temperature sensors are unavailable
        
        # Check for CPU temperature (you can customize for GPU if needed)
        if 'coretemp' in sensors:  # Common on many systems
            return max(temp.current for temp in sensors['coretemp'])
        elif 'acpitz' in sensors:  # Alternative key for laptops
            return max(temp.current for temp in sensors['acpitz'])
        return None

    def save_model_state(state, save_path):
        """Helper function to save model state."""
        # Ensure the parent directory exists
        save_dir = Path(save_path).parent
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(save_dir, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {save_dir}")

        # Save the state
        torch.save(state, save_path)
        print(f"Model state saved to {save_path}")

    def train(self, train_data_path, val_data_path=None, epochs=1000, batch_size=6, log_interval=20, temp_threshold = 90):
        # Prepare dataset and dataloaders
        train_dataset = self.load_and_tokenize_data(train_data_path)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # If validation data is provided
        if val_data_path:
            val_dataset = self.load_and_tokenize_data(val_data_path)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # Track best model based on validation loss
        best_val_loss = float('inf')
        best_train_loss = float('inf')  # Initialize best_train_loss to infinity
        best_model_path = None
        no_improvement_epochs = 0

        # Load checkpoint if available
        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # print(f"Model loaded from checkpoint: {checkpoint_path}")
            try:
                # Try loading the model state_dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from checkpoint: {checkpoint_path}")
            except KeyError:
                print(f"Model state_dict not found in checkpoint {checkpoint_path}. Initializing from scratch.")
                self.model.load_state_dict(checkpoint)

            try:
                # Try loading the optimizer state_dict
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer loaded from checkpoint.")
            except KeyError:
                print("Optimizer state_dict not found in checkpoint. Initializing optimizer from scratch.")

            try:
                # Try loading the scheduler state_dict
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler loaded from checkpoint.")
            except KeyError:
                print("Scheduler state_dict not found in checkpoint. Initializing scheduler from scratch.")
            try:
                # Load epoch if available
                self.epoch = checkpoint['epoch']
                print(f"Resuming from epoch {self.epoch}")
            except KeyError:
                self.epoch = 0
                print("No epoch information found in checkpoint. Starting from epoch 0.")

        else:
            print("No checkpoint found. Starting training from scratch.")
            self.epoch = 0
            # Initialize optimizer and scheduler from scratch if no checkpoint

        # Start training loop
        for epoch in range(self.epoch + 1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")
            
            # Track loss and predictions for metrics calculation
            epoch_loss = 0.0
            epoch_labels = []
            epoch_preds = []

            self.model.train()

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", enabled=self.scaler is not None):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                if self.scaler:
                    # Mixed Precision
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard Training
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
                    self.optimizer.step()

                # Calculate gradient norm
                grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                # Update loss and predictions
                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                epoch_labels.extend(labels.cpu().numpy())
                epoch_preds.extend(preds.cpu().numpy())

                # Calculate batch-level metrics
                accuracy = accuracy_score(epoch_labels, epoch_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(epoch_labels, epoch_preds, average='weighted', zero_division=0)

                # Log batch metrics
                if batch_idx % log_interval == 0:
                    current_lr = self.get_lr()
                    log_data = {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "loss": loss.item(),
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "learning_rate": current_lr,
                        "gradient_norm": grad_norm
                    }
                    self.log_to_json(log_data, "logs/train_log.json")
                    print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, "
                        f"Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, "
                        f"Recall = {recall:.4f}, F1 = {f1:.4f}, "
                        f"Gradient Norm = {grad_norm:.4f}, LR = {current_lr:.6e}")
                    
                self.scheduler.step()

                current_step = self.scheduler.last_epoch  # This is the step number (last_epoch)
                current_lr = self.optimizer.param_groups[0]['lr']  # Get current learning rate
                print(f"Epoch {epoch}, Batch {batch_idx}, Step {current_step}, LR: {current_lr}")

            # End of training loop for epoch
            avg_loss = epoch_loss / len(train_dataloader)
            accuracy = accuracy_score(epoch_labels, epoch_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(epoch_labels, epoch_preds, average='weighted')
            cm = confusion_matrix(epoch_labels, epoch_preds)
            mcc = matthews_corrcoef(epoch_labels, epoch_preds)

            # Log epoch-level metrics
            epoch_log_data = {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "mcc": mcc,
                "confusion_matrix": cm.tolist()
            }
            self.log_to_json(epoch_log_data, "logs/train_log.json")
            print(f"Epoch {epoch} complete. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")

            # Validation phase (if validation data is provided)
            val_loss, val_accuracy = None, None
            if val_data_path:
                try:
                    val_loss, val_accuracy = self.validate(val_dataloader)
                    print(f"Validation - Epoch {epoch}: Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.4f}")
                except Exception as e:
                    print(f"Validation failed for epoch {epoch}. Error: {str(e)}")
                    # Optionally, log validation failure
                    log_data = {
                        "epoch": epoch,
                        "validation_error": str(e)
                    }
                    self.log_to_json(log_data, "logs/train_log.json")

            # Save best model based on validation loss
            if val_data_path and val_loss is not None:
                try:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'loss': val_loss
                        }, best_model_path)
                        print(f"New best model saved to {best_model_path}")
                        no_improvement_epochs = 0  # Reset no improvement counter
                    else:
                        print(f"Epoch {epoch} did not improve. No checkpoint saved.")
                        no_improvement_epochs += 1
                    # Handle the case where saving the full checkpoint fails
                except:
                    print(f"Error saving checkpoint: {e}")
                    if val_loss < best_val_loss:
                        best_train_loss = avg_loss
                        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
                        # Save only the model state dict if the full checkpoint saving fails
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f"Model checkpoint saved to {checkpoint_path}")
            if val_data_path and val_loss is not None:
                try:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'loss': val_loss
                        }, best_model_path)
                        print(f"New best model saved to {best_model_path}")
                        no_improvement_epochs = 0  # Reset no improvement counter
                    else:
                        print(f"Epoch {epoch} did not improve. No checkpoint saved.")
                        no_improvement_epochs += 1
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
                        # Save only the model state dict if the full checkpoint saving fails
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f"Model checkpoint saved to {checkpoint_path}")
            else:
                try:
                    if avg_loss < best_train_loss:
                        best_train_loss = avg_loss
                        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'loss': avg_loss
                        }, checkpoint_path)
                        print(f"Checkpoint saved to {checkpoint_path}")
                        no_improvement_epochs = 0  # Reset no improvement counter
                    else:
                        print(f"Epoch {epoch} did not improve. No checkpoint saved.")
                        no_improvement_epochs += 1
                except Exception as e:
                    # Handle the case where saving the full checkpoint fails
                    print(f"Error saving checkpoint: {e}")
                    if avg_loss < best_train_loss:
                        best_train_loss = avg_loss
                        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch}.pth"
                        # Save only the model state dict if the full checkpoint saving fails
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f"Model checkpoint saved to {checkpoint_path}")
            # Check if no improvement after 2 epochs, then shut down
            if no_improvement_epochs >= 2:
                print(f"Model did not improve for {no_improvement_epochs} epochs. Shutting down...")
                self.shutdown()
        self.shutdown()

    def validate(self, val_data_path, batch_size=6):
        """
        Perform a single validation epoch on the provided validation dataset.

        Parameters:
        - val_data_path (str): Path to the validation dataset.
        - batch_size (int): Number of samples per batch for validation.

        Returns:
        - avg_loss (float): Average loss over the validation dataset.
        - accuracy (float): Accuracy over the validation dataset.
        """
        start_time = time.time()
        val_dataset = self.load_and_tokenize_data(val_data_path)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)

        checkpoint_path = "checkpoints/checkpoint_epoch_4.pth" # Or other checkpoint you're loading
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        print(f"Model loaded from checkpoint: {checkpoint_path}")
        self.model.eval()  # Set model to evaluation mode
        epoch_loss = 0.0
        epoch_labels = []
        epoch_preds = []

        with torch.no_grad():  # Disable gradient computations
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                # Update loss and predictions
                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                epoch_labels.extend(labels.cpu().numpy())
                epoch_preds.extend(preds.cpu().numpy())

        # Compute metrics
        avg_loss = epoch_loss / len(val_dataloader)
        accuracy = accuracy_score(epoch_labels, epoch_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(epoch_labels, epoch_preds, average='weighted')
        cm = confusion_matrix(epoch_labels, epoch_preds)

        # Log validation metrics
        val_log_data = {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist()
        }
        self.log_to_json(val_log_data, "logs/val_log.json")

        # Print validation results
        print(f"\nValidation complete. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Time elapsed {time.time() - start_time} seconds")
        return avg_loss, accuracy



def main():
    model = ClassificationModel("microsoft/deberta-large") # Loads with my checkpoint in the script from my local repository. Please adjust script accordingly to load with HF models. 
    model.train(train_data_path="data/train/train_data.pt")  # Keep in mind these are invalid paths copied from my local repository because my virtual repo. doesn't have the space to store them.
    model.validate(val_data_path="data/val/val_data.pt")


if __name__ == '__main__':
    main()
