#!/usr/bin/env python
"""
Simplified blockchain implementation for federated learning model tracking
"""
import hashlib
import json
import time
from datetime import datetime

class Block:
    """Simple block implementation"""
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
        
    def calculate_hash(self):
        """Calculate hash of the block contents"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": str(self.data),
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    """Simple blockchain implementation"""
    def __init__(self):
        """Initialize the blockchain with the genesis block"""
        self.chain = [self.create_genesis_block()]
        
    def create_genesis_block(self):
        """Create the first block in the chain"""
        return Block(0, datetime.now().isoformat(), "Genesis Block", "0")
    
    def get_latest_block(self):
        """Get the most recent block in the chain"""
        return self.chain[-1]
    
    def add_block(self, data):
        """Add a new block to the chain"""
        index = len(self.chain)
        timestamp = datetime.now().isoformat()
        previous_hash = self.get_latest_block().hash
        
        new_block = Block(index, timestamp, data, previous_hash)
        self.chain.append(new_block)
        
        return new_block
    
    def is_chain_valid(self):
        """Validate the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Check if the current block hash is correct
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check if the previous hash reference is correct
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def print_chain(self):
        """Print the blockchain"""
        for block in self.chain:
            print(f"Block #{block.index}")
            print(f"Timestamp: {block.timestamp}")
            print(f"Data: {block.data}")
            print(f"Hash: {block.hash}")
            print(f"Previous Hash: {block.previous_hash}")
            print("-" * 50)

# Example usage with federated learning
def record_fl_round(blockchain, round_num, global_parameters, metrics):
    """Record federated learning round in the blockchain"""
    # Simplify parameters for storage
    simplified_params = {
        "coef_mean": float(np.mean(global_parameters["coef"])),
        "coef_shape": global_parameters["coef"].shape,
        "intercept": float(global_parameters["intercept"])
    }
    
    # Create data to store
    data = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "parameters_summary": simplified_params,
        "metrics": metrics
    }
    
    # Add to blockchain
    new_block = blockchain.add_block(data)
    
    return new_block
