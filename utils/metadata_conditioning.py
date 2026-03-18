# Import libraries

import torch
import torch.nn as nn

# Encode HAM10000 metadata into a 768-dimensional vector suitable for Stable Difussion cross-attention conditioning

class MetadataConditionEncoder(nn.Module):

    def __init__(
        self,
        num_dx: int,
        num_sites: int,
        num_sex: int,
        hidden_dim: int = 128,
        final_dim: int = 768
    ):

        super().__init__()

        # Embeddings for categorical metadata
        
        self.dx_emb = nn.Embedding(num_dx, 32)
        self.site_emb = nn.Embedding(num_sites, 32)
        self.sex_emb = nn.Embedding(num_sex, 16)

        # Small MLP for continuous age feature
        
        self.age_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16)
        )

        # Combine everything 
        
        self.fc = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, final_dim)
        )
        

    def forward(self, meta_batch):

        dx = self.dx_emb(meta_batch["dx"])      
        site = self.site_emb(meta_batch["site"]) 
        sex = self.sex_emb(meta_batch["sex"])    

        age = meta_batch["age"].unsqueeze(-1)    
        age = self.age_mlp(age)                  

        concat = torch.cat([dx, site, sex, age], dim=-1)  

        return self.fc(concat)  

