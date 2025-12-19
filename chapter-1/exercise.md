**Exercise 1.1: \[Self-Play\] Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself, with both sides learning. What do you think would happen in this case? Would it learn a different policy for selecting moves?**

What would happen: It tends to push the agent toward an unexploitable (minimax-like) policy, since both sides adapt and remove weaknesses. This differs from policies learned against random opponents, which often exploit mistakes that disappear under stronger play.

Different policy: Yes, a policy learned against a random opponent exploits mistakes. A risk move might be taken to set up a trap, knowing the random player won’t block it.

 A policy that is optimal against a random player is rarely optimal against a learning/perfect player.

**Exercise 1.2: \[Symmetries\] Many tic-tac-toe positions appear different but are really the same because of symmetries. How might we amend the learning process described above to take advantage of this? In what ways would this change improve the learning process? Now think again. Suppose the opponent did not take advantage of symmetries. In that case, should we? Is it true, then, that symmetrically equivalent positions should necessarily have the same value?**

How to amend: We can merge symmetrically duplicated states. For example, we transform the board to a canonical form before doing value look up.

What ways would improve: Leverage symmetric property accelerates learning because agents will simultaneously learn all symmetric variations of states.

Should we use it: If two states are truly symmetric under the game rules, then their optimal values are identical, regardless of whether the opponent happens to exploit symmetry.

**Exercise 1.3: \[Greedy Play\] Suppose the reinforcement learning player was greedy, that is, it always played the move that brought it to the position that it rated the best. Might it learn to play better, or worse, than a nongreedy player? What problems might occur?**

Likely worse because it suffers from a lack of exploration. For example, the agent stumbles upon a move that leads to a draw early on, it then rates that move highly and sticks to play that move forever, losing the opportunity to discover a slightly riskier move that actually leads to a win. It gets stuck on a suboptimal policy.

**Exercise 1.4: \[Learning from Exploration\] Suppose learning updates occurred after all moves, including exploratory moves. If the step-size parameter is appropriately reduced over time (but not the tendency to explore), then the state values would converge to a different set of probabilities. What (conceptually) are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves? Assuming that we do continue to make exploratory moves, which set of probabilities might be better to learn? Which would result in more wins?**

Two sets of probabilities:

1. Learn from exploratory moves. The values represent the probability of winning if we continue to play with the current strategy.
2. Not learn. The value represents the probability of winning if we play optimally from now on.

Which is better: The second because exploration is a learning artifact, not a behavior we want long-term. It leads to stronger greedy policies and more wins once learning stabilizes.

Which result in more wins: the second.

**Exercise 1.5: \[Other Improvements\] Can you think of other ways to improve the reinforcement learning player? Can you think of any better way to solve the tic-tac-toe problem as posed?**

Possible improvements:

- Look a few more steps ahead instead of just one step.
- Use a neural network or other models to estimate the value of a state.
- Update all moves in the sequence instead of only updating the move immediately preceding the result.
