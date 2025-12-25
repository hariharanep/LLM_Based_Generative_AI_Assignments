# Importing all the required libraries/dependencies
import torch
import torch.nn as nn
from dataclasses import dataclass
import time
import os
import argparse
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, Schedule1F1B, ScheduleInterleaved1F1B

# global variables taken from the tutorial code
global rank, device, pp_group, stage_index, num_stages, start_time, end_time

# init_distributed method is the exact same as the tutorial code
def init_distributed():
   global rank, device, pp_group, stage_index, num_stages

   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
   dist.init_process_group()

   pp_group = dist.new_group()
   stage_index = rank
   num_stages = world_size

# ModelArgs class is the exact same as the tutorial code with n_processes(number of processes) attribute added to it
@dataclass
class ModelArgs:
   dim: int = 504
   n_layers: int = 8
   n_heads: int = 8
   vocab_size: int = 10000
   n_processes: int = 2

# Transformer Model class is the exact same as the tutorial code
class Transformer(nn.Module):
   def __init__(self, model_args: ModelArgs):
      super().__init__()

      self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

      self.layers = torch.nn.ModuleDict()
      for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dim, model_args.n_heads, batch_first=True)

      self.norm = nn.LayerNorm(model_args.dim)
      self.output = nn.Linear(model_args.dim, model_args.vocab_size)

   def forward(self, tokens: torch.Tensor):
      h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

      for layer in self.layers.values():
            h = layer(h, h)

      h = self.norm(h) if self.norm else h
      output = self.output(h).clone() if self.output else h
      return output

# This tracer_model_split function splits the model into stages depending on the number of processes
def tracer_model_split(model, example_input, num_processes):
    split_spec = {}
    
    if num_processes == 2:
        # Split model into two stages
        # First Stage: token embeddings layers plus first half of the TransformerDecoder layers
        # Second Stage: second half of the TransformerDecoder layers plus normalization and output layer
        halfway = model_args.n_layers // 2
        split_spec = {
            f"layers.{halfway}": SplitPoint.BEGINNING
        }        
    elif num_processes == 4:
        # Split model into four stages
        # First Stage: token embeddings layers plus first quarter of the TransformerDecoder layers
        # Second Stage: second quarter of the TransformerDecoder layers
        # Third Stage: third quarter of the TransformerDecoder layers
        # Fourth Stage: last quarter of the TransformerDecoder layers plus normalization and output layer
        split = model_args.n_layers // 4
        split_spec = {
            f"layers.{split}": SplitPoint.BEGINNING,      
            f"layers.{2*split}": SplitPoint.BEGINNING,   
            f"layers.{3*split}": SplitPoint.BEGINNING
        }

    # Returns stage in this pipeline that corresponds to the current stage_index    
    stage = pipeline(
        model,
        mb_args=(example_input,),  
        split_spec=split_spec,
    ).build_stage(
        stage_index,
        device,
    )    
    return stage

# This create_interleaved_stages_tracer function splits the model into interleaved stages for Interleaved1F1B scheduling
# num_chunks represents the number of chunks/stages per rank
def create_interleaved_stages_tracer(model, example_input, num_processes, num_chunks):
    # Total chunks in total
    total_chunks = num_processes * num_chunks
    # Total number of model layers each chunk should store
    layers_per_chunk = model_args.n_layers // total_chunks
    
    # Generate all the split points for each chunk
    split_spec = {}
    for i in range(1, total_chunks):
        split_layer = i * layers_per_chunk
        split_spec[f"layers.{split_layer}"] = SplitPoint.BEGINNING

    # Create pipeline for Interleaved1F1B scheduling
    pipe = pipeline(
        model,
        mb_args=(example_input,),
        split_spec=split_spec,
    )

    """
    Interleaves the stages and returns the result as a list of stages for this rank
    Example for num_processes = 2, num_chunks = 2, 4 total stages(Stage 0, Stage 1, Stage 2, Stage 3)
    Rank = 0, first process
    chunk_idx = 0: global_stage_idx = 0 + (0 * 2) = 0 = Stage 0
    chunk_idx = 1: global_stage_idx = 0 + (1 * 2) = 2 = Stage 2

    Rank = 1, second process
    chunk_idx = 0: global_stage_idx = 1 + (0 * 2) = 1 = Stage 1
    chunk_idx = 1: global_stage_idx = 1 + (1 * 2) = 3 = Stage 3
    """    
    stages = []
    for chunk_idx in range(num_chunks):
        global_stage_idx = rank + (chunk_idx * num_processes)
                
        stage = pipe.build_stage(global_stage_idx, device)
        stages.append(stage)
    
    # Return list of stages that correspond to the current rank
    return stages


if __name__ == "__main__":
    # same as tutorial code
    init_distributed()

    # parse the additional model and distributed parallelism arguments provided in command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int)
    parser.add_argument("--attention_heads", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--schedule", type=str)
    # num_chunks will always be passed in as half of layers(total number of TransformerDecoder layers) to distribute the workload evenly across all ranks/GPUs
    parser.add_argument("--num_chunks", type=int, default=2)

    args = parser.parse_args()

    # Set model arguments per the model and distributed parallelism arguments provided in command line
    model_args = ModelArgs()
    model_args.n_layers = args.layers
    model_args.n_heads = args.attention_heads
    model_args.n_processes = args.processes
    num_processes = args.processes

    # Create model
    model = Transformer(model_args=model_args)

    # Random singular batch of dummy data
    x = torch.ones(32, 500, dtype=torch.long)
    y = torch.randint(0, model_args.vocab_size, (32, 500), dtype=torch.long)

    # Splitting dummy data into 4 micromatches
    num_microbatches = 4
    example_input_microbatch = x.chunk(num_microbatches)[0]

    # use tracer_model_split to get stage for GPipe and 1F1B schedules
    stage = None
    if args.schedule in ["GPipe", "1F1B"]:
        stage = tracer_model_split(model, example_input_microbatch, num_processes)
        stages = None
    # use create_interleaved_stages_tracer to get list of stages for this rank for Interleaved1F1B schedule
    elif args.schedule == "Interleaved1F1B":
        stages = create_interleaved_stages_tracer(model, example_input_microbatch, num_processes, args.num_chunks)
        stage = None

    # Move data to device
    x = x.to(device)
    y = y.to(device)

    # tokenwise_loss_fn taken from tutorial code
    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_args.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    schedule = None
    optimizer = None
    # Create schedule and Adam optimizer with stage or stages list created previously
    # GPipe and 1F1B schedules are 1 stage per rank
    # Interleaved1F1B schedule is multiple stages per rank hence why its handled differently
    if args.schedule == "GPipe":
        schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
        optimizer = torch.optim.Adam(stage.submod.parameters(), lr=0.001)
    elif args.schedule == "1F1B":
        schedule = Schedule1F1B(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
        optimizer = torch.optim.Adam(stage.submod.parameters(), lr=0.001)
    elif args.schedule == "Interleaved1F1B":
        schedule = ScheduleInterleaved1F1B(stages, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
        params = []
        for s in stages:
            params.extend(s.submod.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
    
    num_epochs = 3

    # Start tracking start time and ending time at one specific rank to avoid invoking  start_time = time.time() for each rank in the beginning
    if rank == num_processes - 1:
        global start_time
        start_time = time.time()
    
    # Train dummy data for 3 epochs
    for epoch in range(num_epochs):
        print(f"Rank: {rank}, Starting Epoch: {epoch + 1}", flush=True)

        # clears the gradients of all parameters that the optimizer is currently tracking
        optimizer.zero_grad()

        # If rank = last rank in schedule, print the avg loss
        if rank == num_processes - 1:
            losses = []
            output = schedule.step(x, target=y, losses=losses)
            avg_loss = sum(losses) / len(losses)
            print(f"Rank: {rank}, Avg Loss: {avg_loss:.4f}", flush=True)
        # Execute stage normally
        else:
            schedule.step(x)
        
        # update model parameters
        optimizer.step()
    
    
    # End tracking for total training time, record total training throughput and approximate end to end training time
    if rank == num_processes - 1:
        global end_time
        end_time = time.time()
        total_time = end_time - start_time
        # Dummy data batch = (32 rows, 500 tokens each row)
        # Trained 3 epochs(went through whole batch 3 times)
        # Total throughput = (32 x 500 x 3) / total end to end training time
        print(f"Total Training Throughput: {((32 * 500 * 3) / total_time):.3f}")
        print(f"End To End Training Time: {(total_time):.3f}")
    
    # cleans up and terminates previously initialized distributed process group
    dist.destroy_process_group()
