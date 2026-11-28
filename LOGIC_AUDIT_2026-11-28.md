# Economic, Mathematical, and Game-Theory Audit — 2026-11-28

**Scope:** `origin/main` at `b893eb3`, the subsistence world
(`worlds/subsistence/`) and the money world (`worlds/money/`), shared
`utils/`, replay data, and the experiment harness.

**Method:** Code reading, targeted runtime probes, and regression tests. The
pre-fix suite passed 240 tests; every item below was reproduced or is enforced by
a new regression test. Post-fix suite: **270 tests pass**.

**Pricing decision:** prices and price beliefs are now **fractional** and are
**not clamped**. Goods, trade quantities, and the berry leg of a trade remain
whole. The residual from a fractional price is an explicit indivisibility cost.
An agent can therefore fail to trade because its opportunity cost is not a whole
number, and a tiny positive interest rate can round up to one whole unit. Those
are accepted consequences of whole-good exchange, not pricing errors.

---

## 1. Debt service could kill the debtor

**Problem:** `_can_pay()` only checked that the debtor owned the full amount.
A borrower with six berries and a six-berry loan repaid all six, then ate
nothing and died. Formal loans also settled after soft gifts, contrary to the
documented order.

**Fix:** berry debt now leaves `FOOD_PER_ITERATION` berries untouched before
service; partial repayment is allowed above that reserve; default seizure also
leaves a living berry debtor's meal; formal loans settle before soft gifts; and
berry-loan repayment uses the reserve consistently.

**Locations:** `worlds/subsistence/features/transfers.py`,
`worlds/subsistence/world.py` (feature registration order),
`worlds/subsistence/MODEL.md`.

**Tests:** `tests/subsistence/test_audit_fixes.py`
(`test_formal_berry_debt_reserves_todays_food`,
`test_default_does_not_seize_todays_food`,
`test_formal_loans_settle_before_soft_gifts`).

---

## 2. Credit demand put food last

**Problem:** a borrower's demand order was crates → wood → berries, and
origination issued at most one loan per borrower per tick. A starving borrower
could borrow crate material or wood while going hungry.

**Fix:** berries are now the first credit demand, followed by crates and wood.
Demand is recomputed at each borrower's turn so an agent that lent goods earlier
in the same tick does not act on stale demand.

**Location:** `worlds/subsistence/features/credit.py`.

**Test:** `test_credit_demand_puts_food_first`.

---

## 3. A low-health borrower with collateral was refused

**Problem:** `_offers()` rejected a low-trust loan when
`available_collateral <= 0 or borrower.health < 2`. The documented rule refuses
a low-health borrower only when there is no collateral.

**Fix:** the health test is removed from the collateral branch. A secured
low-health borrower can now borrow food.

**Location:** `worlds/subsistence/features/credit.py`.

**Test:** `test_low_health_borrower_can_pledge_collateral_for_food`.

---

## 4. The market allowed self-trades and two-sided agents

**Problem:** `_match_market()` had no `seller != buyer` guard. A crafted case
produced a trade where the same agent was both sides; goods netted to zero but
trade volume, prices, and beliefs changed. `_wood_market_plan()` could also list
one agent as both buyer and seller.

**Fix:** self-matches are skipped, and an agent that is listed as a seller of
wood is not also listed as a buyer in the same tick.

**Location:** `worlds/subsistence/world.py`.

**Tests:** `test_market_does_not_allow_self_trades`;
`test_market_prices_are_fractional_but_berry_payments_are_whole`.

---

## 5. Pledged crates were not encumbered

**Problem:** crate surplus and credit supply used `crate_count` without
subtracting `pledged_crates()`. A borrower could sell or lend the crate that
secured another loan, leaving the lender with no collateral.

**Fix:** crate market surplus uses unpledged crates; credit supply keeps
`max(LENDER_CRATE_RESERVE, pledged)` crates out of the lendable stock.

**Locations:** `worlds/subsistence/world.py`,
`worlds/subsistence/features/credit.py`.

**Test:** `test_pledged_crates_cannot_be_sold_or_lent`.

---

## 6. Crate durability was ignored in prices and collateral

**Problem:** a crate at 5% durability and a new crate had the same sale price
and the same collateral value. Used crates were perfect substitutes for new
ones.

**Fix:** crate sale prices, collateral values, and credit coverage scale with
the durability of the crates actually offered or seized. Crate transfers move
lowest-durability crates to a buyer/lender/repayer and highest-durability crates
to a seizing creditor.

**Locations:** `worlds/subsistence/world.py`,
`worlds/subsistence/features/transfers.py`,
`worlds/subsistence/features/credit.py`.

**Test:** `test_crate_sale_price_depends_on_durability`.

---

## 7. Crate collateral was valued inconsistently and could over-seize

**Problem:** a crate was valued at 8 wood or 10 berries as collateral while its
market opportunity cost could be hundreds of berries. A 2-wood default seized a
whole crate, recorded `recovered=2`, `loss=0`, and hid the excess transfer.

**Fix:** one common berry-equivalent crate value is used, converted to wood at
the current market price; durability scales it; seizure prefers the highest
durability crate; and any excess over the claim is recorded as
`collateral_excess` (also exposed as `collateral_excess_value`). Market mode now
prices crate-loan durability risk instead of hard-coding zero interest.

**Locations:** `worlds/subsistence/world.py`,
`worlds/subsistence/features/transfers.py`,
`worlds/subsistence/features/credit.py`, replay event schema.

**Accepted residual:** a crate is indivisible, so a claim smaller than one crate
can still receive a whole crate. The excess is now explicit rather than reported
as zero loss.

---

## 8. Creditor death was booked as a borrower default

**Problem:** subsistence metrics counted a creditor-death write-off as a
default and default loss. The money world even observed a repayment failure and
lowered the borrower's reputation when the creditor had died.

**Fix:** write-offs are a separate `write_off` event with
`written_off_*` metrics; they do not increment defaults, default loss, or
borrower reputation. The money world observes repayment only when both parties
are alive.

**Locations:** `worlds/subsistence/features/transfers.py`,
`worlds/subsistence/features/credit.py`, `worlds/subsistence/replay.py`,
`worlds/money/world.py`.

**Tests:** `test_creditor_death_is_not_observed_as_borrower_default`;
subsistence credit tests for write-off classification.

---

## 9. Soft-gift partial repayments over-awarded trust

**Problem:** `settle_soft_gifts()` awarded `moved / principal`, bypassing
`TRUST.GAIN`. A four-berry gift repaid in one-berry steps added up to 1.0 trust;
the configured gain was 0.08.

**Fix:** the award is `TRUST.GAIN * moved / principal`, so a fully repaid gift
adds at most one configured gain, and partial repayments add proportionally.

**Locations:** `worlds/subsistence/features/transfers.py`,
`worlds/subsistence/features/trust.py`.

**Test:** `test_soft_gift_trust_gain_uses_configured_gain`.

---

## 10. Crate trades could set a non-wood headline price

**Problem:** crate prices were divided by `wood_per_crate` and fed into the wood
price and wood beliefs. A crate trade at 322 with eight wood per crate produced
a headline “wood price” of 40. The old code then depended on `MAX_PRICE` to hide
this.

**Fix:** wood prices and wood beliefs come only from wood trades. Crate prices
are recorded separately in `last_crate_price`. There is no price clamp; the
market is allowed to discover any positive fractional ratio.

**Location:** `worlds/subsistence/world.py`.

**Test:** `test_market_price_is_not_clamped` and the fractional-price test.

---

## 11. Soft-gift cap created free gifts and had no default cost

**Problem:** once a receiver was at `MAX_OUTSTANDING`, later gifts still moved
but created no obligation, so the cap turned reciprocal gifts into free
transfers. A living receiver could also keep a soft claim forever with no trust
penalty.

**Fix:** the gift amount is capped before it moves by the receiver's remaining
obligation room, so no free excess gift is created. A soft obligation older than
`RECIPROCITY.DEFAULT_AFTER_DAYS` is written off, penalises trust, and frees the
cap. Setting that value to 0 keeps the old indefinite-claim behaviour.

**Locations:** `utils/social.py`,
`worlds/subsistence/features/gifting.py`,
`worlds/subsistence/features/reciprocity.py`,
`worlds/subsistence/features/transfers.py`.

**Tests:** `test_soft_gift_cap_does_not_create_free_gifts`,
`test_soft_gift_default_penalises_trust_after_age_limit`.

---

## 12. Credit capacity and interest rounding were misleading

**Problem:** crate repayment capacity returned at least one crate even when a
borrower produced almost no wood or had no craft time. Repayment used `ceil`, so
a one-unit loan at any positive rate doubled, and the realised rate could exceed
`MAX_INTEREST`.

**Fix:** crate capacity now requires both wood and craft time; berry capacity is
net of food and includes wood income; wood capacity remains production-based.
`ceil` is retained deliberately because repayment is whole goods, and the docs
now state that a small positive quoted rate can round up to one whole unit. This
is the accepted indivisibility cost requested for this model.

**Locations:** `worlds/subsistence/features/credit.py`,
`worlds/subsistence/MODEL.md`.

**Test:** `test_crate_credit_capacity_requires_real_crate_production`.

---

## 13. Seller reservation prices moved the wrong way with hunger

**Problem:** a hungry seller with health 3 raised its ask by one; crate sellers
raised their ask by two whenever health was below maximum and never had the
desperate-seller override. The documented rule says a hungry seller accepts
almost any price.

**Fix:** hunger now lowers the seller's ask; a near-starving seller goes to the
positive-price floor; the subsistence floor applies only when the seller is not
hungry. Crate seller prices also scale with the durability of the offered
crates.

**Location:** `worlds/subsistence/world.py`.

**Tests:** `test_crate_sale_price_depends_on_durability`; existing market tests.

---

## 14. Exploration could make an agent do worthless work

**Problem:** epsilon-greedy exploration swapped berry and wood values even when
one activity had zero value. With storage or spoilage disabled, an agent could
spend the day chopping wood that had no use.

**Fix:** exploration swaps only when both activities have positive value.

**Location:** `worlds/subsistence/world.py`.

**Tests:** existing `test_wood_is_not_a_specialisation_without_storage` and
`..._without_spoilage`.

---

## 15. Experiments reported final-tick flows as run outcomes

**Problem:** `run_experiment()` read `metrics_history[-1]`. Most subsistence
metrics are per-tick flows, so a 200-tick run with 72 loans and 40 repayments
reported `loans_originated=0`, `loans_repaid=0`, and `loans_defaulted=0` if the
last tick was quiet.

**Fix:** the harness now aggregates flow metrics over the run, averages rate and
intensity metrics, and leaves stock/cumulative metrics at the final value. This
is shared by both worlds.

**Locations:** `utils/experiments.py`,
`worlds/subsistence/experiments.py`.

**Tests:** `test_flow_metrics_are_summed_across_the_run`.

---

## 16. Confidence intervals were too narrow for small seed counts

**Problem:** every group used `1.96 * std / sqrt(n)`. With one seed the interval
had zero width; with two or three seeds it was far too narrow.

**Fix:** two-sided 95% Student-t critical values are used, and a one-seed
interval is `NaN` (undefined) instead of a false zero-width interval.

**Location:** `utils/experiments.py`.

**Tests:** `test_single_seed_interval_is_undefined`,
`test_small_seed_interval_uses_student_t`.

---

## 17. Replay metadata recorded end-of-run values as “initial”

**Problem:** `replay.py` read the live `agent.crate_count` and
`agent.price_belief` for `initial_crates` and `price_belief`. After a run those
were final values. Tick zero also hard-coded `storage_used=0` despite starter
berries in crates.

**Fix:** metadata is read from the iteration-zero snapshot, and tick-zero
storage use/utilization is computed from the initial states.

**Location:** `worlds/subsistence/replay.py`.

**Test:** `test_replay_metadata_uses_initial_crates_and_beliefs`.

---

## 18. `specialisation` ignored crafting

**Problem:** `abs(labor_berries - 0.5) * 2` treated all non-berry time as wood
specialisation, including all-day crafting.

**Fix:** it is a normalised Herfindahl index over berry, wood, and craft labour:
0 for an even three-way split, 1 for a single activity.

**Location:** `worlds/subsistence/world.py`.

**Documentation:** `worlds/subsistence/MODEL.md`.

---

## 19. Re-initializing a world preserved an active shock

**Problem:** `initialize_agents()` cleared logs but not
`ShockLayer.active_until`. A fresh run could start mid-drought with an empty
shock log.

**Fix:** feature layers have a `reset()` hook; shock state and regeneration
factors are cleared on re-initialization.

**Locations:** `worlds/subsistence/world.py`,
`worlds/subsistence/features/base.py`,
`worlds/subsistence/features/shocks.py`.

**Test:** `test_reinitializing_a_world_clears_active_shocks`.

---

## 20. Zero iterations or zero agents crashed

**Problem:** `--iterations 0` and `run_experiment(0)` hit
`metrics_history[-1]`. `AGENTS=0` hit an empty Gini/DataFrame error.

**Fix:** experiment iterations must be at least one; the CLIs print “no
iterations run” instead of indexing an empty history; both worlds require at
least one agent.

**Locations:** `utils/experiments.py`,
`worlds/subsistence/cli.py`, `worlds/subsistence/world.py`,
`worlds/money/world.py`.

**Tests:** `test_zero_iterations_are_rejected`,
`test_zero_agents_is_rejected`.

---

## 21. Money-world config validation gaps

**Problem:** negative `LIVING_COST` drove the commons fund negative;
`REPAY_FRACTION=1.5` repaid 15 on a 10-unit debt; negative resources/floors,
reversed ranges, negative memory, and zero agents were unchecked.

**Fix:** construction now validates agents, iterations, initial resources,
floor, living cost, giving/receiving floors, affordability scores, generosity/
acceptance/risk ranges, memory size, transfer bounds, and social settings.
`REPAY_FRACTION` is constrained to `[0, 1]`. The living-cost clamp no longer
raises an agent to the floor, so it cannot create money.

**Locations:** `worlds/money/world.py`, `worlds/money/config.yaml`.

**Tests:** `tests/test_audit_money.py`, updated `tests/test_social.py`,
`tests/test_main.py`.

---

## 22. Money-world adaptation and information-shock recovery

**Problem:** enabling shocks with an empty schedule and no market collapsed
risk tolerance from 0.5 to 0. An information-only shock returned
`recovery_time=0` even though it does not move unmet demand.

**Fix:** adaptation runs only while a scarcity shock is active and the market is
enabled. `recovery_time()` returns `None` for information shocks.
`mean_risk_tolerance` now uses living agents only.

**Location:** `worlds/money/world.py`.

**Tests:** `tests/test_shocks.py`
(`test_empty_shock_layer_does_not_adapt`,
`test_information_shock_has_no_demand_recovery_time`).

---

## 23. Money-world acceptance ignored the debt it created

**Problem:** acceptance used only affordability, even though accepting a
reciprocal gift creates a debt with late fees and reputation loss. The docs
called acceptance “weakly dominant” while also saying default had a real cost.

**Fix:** when reciprocity is enabled, existing debt reduces acceptance
probability; the docs now describe acceptance as a debt-aware heuristic rather
than a free action.

**Location:** `worlds/money/world.py`, `worlds/money/MODEL.md`.

**Test:** `test_existing_debt_reduces_acceptance_probability`.

---

## 24. Failed-order belief updates were unbounded

**Problem:** after removing price clamps, a failed buyer raised its belief by one
every tick even when the bid was already capped by its berry budget, and a failed
seller lowered its belief below its own subsistence floor. Mean beliefs drifted
to 50–160 berries per wood while realised trades were near 9.

**Fix:** a failed buyer updates only while the belief is below its affordable
budget; a non-hungry failed seller stops lowering once the ask reaches its
subsistence floor. Beliefs remain fractional and unclamped, but they no longer
random-walk away from a feasible order.

**Location:** `worlds/subsistence/world.py`.

**Probe:** default seed 42 over 1000 ticks now ends with mean belief about 6.4
and realised price about 7.4, instead of a mean belief above 50.

---

## 25. Stale documentation and UI

**Problem:** the config header, world docstring, and viewer tick-zero story said
villagers start with nothing; they start with ten berries and one crate. The
model doc said one crate holds a full twelve-berry reserve, but the effective
target is `min(10, 12) = 10`. Several price passages still said integer prices
and clamped beliefs.

**Fix:** updated `worlds/subsistence/config.yaml`,
`worlds/subsistence/world.py`, `worlds/subsistence/viewer/viewer.js`,
`worlds/subsistence/MODEL.md`, `worlds/subsistence/README.md`, and the root
`README.md`. The viewer also formats fractional prices and shows `write_off` and
collateral-excess events.

---

## Accepted whole-good inefficiencies

These are deliberate, not bugs:

1. **Fractional prices, whole goods.** The berry leg is rounded to a whole berry,
   so realised terms can differ slightly from the quoted ratio.
2. **Positive interest can round up.** A one-unit loan with a small positive
   quoted rate may repay two units. This is the whole-goods equivalent of an
   integer price grid.
3. **Crate collateral is indivisible.** A claim smaller than a crate can receive
   a whole crate; the excess is recorded as `collateral_excess` instead of being
   hidden as zero loss.
4. **A crate loan repays a whole crate.** Durability is priced, but a worn crate
   cannot be split to match a fractional claim.

---

## Verification

```bash
python -m pytest -q
# 270 passed
```
