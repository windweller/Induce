List of todos for interpretability:
1. Normal text encoder with easy access to important parts.
2. Add data loader for SST (prep for MR, CR)
3. Will be able to interpret models like InferSent and DisSent
4. Add neural rationale model
5. Add neural concrete model  
6. Add Murdoch's code

List of todos for Interactive learning:
1. be able to output the patterns and allow humans to edit (load list, output list)
2. Models need to be able to load in and train on such pattern list
3. Need to be able to evaluate on a given dataset, and observe performance (hopefully improvement)

Think about what the structure should be: let's iron out the moving parts first

- model: there are LSTM (diff configs), CNN, Attention-is-all
- interpretation: multiple interpretation kernel for each model
- visualization (can be shared...this is a common structure -- send in key words, weights, etc.)
- retraining -- we can have multiple re-training strategy, model-specific

So it seems that the best way to organize is to organize by model...