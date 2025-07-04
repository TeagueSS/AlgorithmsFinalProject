# Advanced Synergy Matchmaker 🕹️🎯  
_Balanced lobbies for role-based shooters, powered by greedy look-ahead + Monte-Carlo magic_

Think of your favorite team based game, Rivals, overwatch etc. 
All of these games have open queue but fail to take into account players preferred role.
Not anymore! Here we are finding balanced team comp within games by trying to pre-pick teams
that would be more balances based on players favorite characters. 

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)


## Why this exists  
Default matchmaking ignores who you _want_ to play and how heroes interact.  
Here we:

1. keep separate role queues (`tank | dps | support`)  
2. model **synergies** (who pops off together) and **counters** (rock-paper-scissors-ish) 
3. pull extra candidates, score every duo, and greedily assign the best pair to the weaker team, repeating until the roster is full 
4. simulate 20 rounds with a Monte-Carlo twist, so match quality isn’t decided by one lucky crit 

End result: ± 35 % higher average match quality over a naive queue in my tests 

## Algorithm
This works by pulling more players than needed for each team,
trying all of the combinations, and then grouping them up to see what synergies work best.
By keeping a list of all the possible synergy teamups we compute what 2 players would work best together for a given role. 
Greedy Algorithm is used in this as we are only really keeping our best 2 pairings as we go. 
Finally we append the best players to the lower MMR team so far,
and then the second best to the other one. This algorithm has look ahead as we are considering 
other parings before making our paring for certain. (We see if we take this player and the enemy
team takes the left over, is that a better paring and as such are in a worse spot by taking that player right now.)



## Key features  
| 🏷️   Where in code |
|---|---|
| Per-character MMR tracking  `Player.update_character_mmr`  |
| Dynamic synergy / counter graphs  `CounterSystem`  |
| Greedy + look-ahead team builder  `_build_optimal_teams` |
| Monte-Carlo match simulator  `MatchSimulator`  |
| Impact-based MMR updates  `MMRUpdater`  |

## Quick-start
```bash
git clone https://github.com/<you>/advanced-synergy-matchmaker.git
cd advanced-synergy-matchmaker
pip install  numpy, pandas, scipy, matplotlib
python matchmaking_system.py # -> this runs 1 000 baseline vs. 1 000 advanced matches
