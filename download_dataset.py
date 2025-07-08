from deepinv.datasets import DIV2K
dataset = DIV2K(root="DIV2K", mode="train", download=True)
print(f"Training images: {len(dataset)}")