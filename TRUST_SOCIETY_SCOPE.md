# Society Is Built on Trust

> **Scope note.** This document describes the trust and reciprocity research
> layer. In the current repository it is exercised by the money world in
> `worlds/money/`. The goods-only subsistence world keeps these mechanisms out
> of its core; gifting already lives in `utils/social.py`, and reciprocity and
> trust are intended to become shared social utilities that both worlds can
> compose.

## Core idea

People navigate limited resources, uncertainty, and dependence on others. Trust
allows them to cooperate despite not knowing other people’s intentions.

This project is an agent-based laboratory for studying how trust, reputation,
altruism, and social information affect cooperation under changing conditions.
It is not intended to predict the real economy.

## First principles

1. Agents need goods and have limited resources.
2. Useful interactions require dependence: trade, help, lending, or cooperation.
3. Agents cannot fully observe one another’s intentions or past behaviour.
4. Trust is a belief about another agent’s likely behaviour, not a moral trait.
5. Reputation is social information about that belief; it can be incomplete,
   delayed, biased, or false.
6. Cooperation has both costs and possible future benefits.
7. Agents are boundedly rational: they use local information and simple rules.
8. The environment changes, so behaviour that works in one situation may fail
   in another.

## Minimal world

- The population is closed: no external money or goods enter the simulation.
- There is one consumable good and one money-like resource.
- Living producers create a fixed amount of goods per time step.
- Living consumers demand a fixed amount of goods per time step.
- Consumers pay producers for goods; money is not valuable by itself.
- Unsold goods may remain as inventory; unmet demand is recorded.
- Altruistic transfers are separate from market transactions and must have a
  giver, a recipient, and an explicit cost.
- Trust and reputation affect decisions, not the conservation of resources.
- Death or inactivity must follow an explicit survival rule; it must not be an
  accidental side effect of a clamp or random resource deletion.

## Agent state

Each agent may have:

- money and goods inventory;
- needs and risk tolerance;
- private memories and beliefs about other agents;
- public reputation;
- social connections and access to information;
- preferences for material payoff, fairness, reciprocity, or helping.

Trust should be represented as a belief such as “the probability that this
agent will reciprocate,” not as a fixed generosity score.

## Interaction loop

1. The environment determines available goods, scarcity, and information.
2. Agents observe limited and possibly noisy social information.
3. Agents choose whether to trade, help, lend, refuse, or exploit.
4. Goods, money, utility, beliefs, and reputation are updated.
5. The simulation records both material and social outcomes.

## Useful questions

- When does reputation improve cooperation?
- When does misinformation create unfair exclusion?
- Does altruism protect agents during scarcity or create exploitation?
- How quickly do agents learn whom to trust?
- Is social capital more valuable when material resources are scarce?
- How resilient is cooperation after deaths, shortages, or information shocks?

## Measures of success

- fulfilled demand and unmet demand;
- survival and material welfare;
- inequality;
- cooperation and successful exchange rates;
- reputation accuracy and calibration;
- information value;
- resilience after environmental shocks;
- whether observed behaviour follows from the stated assumptions.

## Implementation order

1. Define money, goods, utility, and conservation rules.
2. Build a minimal market with production, consumption, inventory, and prices.
3. Add private memory and noisy public reputation.
4. Replace fixed behaviour probabilities with simple belief-based decisions.
5. Add altruism and reciprocity as a separate social layer.
6. Add environmental shocks and adaptation.
7. Run controlled experiments across seeds and parameter ranges.

## Non-goals

- predicting real-world economic outcomes;
- modelling every institution or market;
- starting with multiple goods, finance, firms, or complex psychology;
- treating one simulation run as evidence of a universal law.

