Hey everyone,

I recently made it to the top 100 finalists of the OpenEnv Hackathon — 36 hours of building at SST Bangalore alongside roughly 800 builders from across the country.

The theme was building RL environments for LLMs: simulation worlds an agent acts in, gets observations from, and gets scored on. The key ingredient is a verifiable task — one that signals better trajectories without a human in the loop.

I built MedChain, a hospital supply chain simulation where the agent plays a procurement coordinator juggling inventory, orders, and budgets across five enterprise systems. Things don't stay stable: a patient surge at the ER, a product recall, a cold storage failure, a supplier going dark. Each breaks whatever plan you had and forces you to reorder, reroute, or escalate on the fly.

What matters is whether the agent can make decisions now that still hold up two rounds later when the consequences land. That's where RL comes in — the model only learns this look-ahead by experiencing the cost of getting it wrong across many episodes. The reward has 10 components, all computed deterministically from simulation state.

More details here: https://www.youtube.com/watch?v=L47ZVn1syAM

The top 15 submissions were impressive. I particularly liked a CAD design environment and a frontend image-to-code environment — both had especially well thought through reward design.

One of the most meaningful experiences I've had. The Scaler team handled the logistics really well, and the onsite event at SST was fantastic. Big thanks to PyTorch and Hugging Face for putting this together.

Check out my submission: https://github.com/nik-55/medchain-openenv-final-round
Hackathon: https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/
