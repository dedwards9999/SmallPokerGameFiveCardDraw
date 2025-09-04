import random
from typing import List, Tuple, Dict


# 5-Card Draw Poker with simple betting AI
# Bankrolls: You and Opponent start with $50.
# Two betting rounds: pre-draw and post-draw. At most one raise per round.
# Card index inputs are 1-based and must be > 0.


# Card and deck utilities
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = "♠♥♦♣"
RANK_VALUE: Dict[str, int] = {r: i for i, r in enumerate(RANKS, start=2)}  # '2'->2, ..., '10'->10, 'A'->14

def make_deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]

def shuffle_deck(deck: List[str]) -> None:
    random.shuffle(deck)

def deal(deck: List[str], n: int) -> List[str]:
    return [deck.pop() for _ in range(n)]

def card_rank(card: str) -> int:
    rank_str = card[:-1]  # support "10" rank
    return RANK_VALUE[rank_str]

def card_suit(card: str) -> str:
    return card[-1]

def hand_to_str(hand: List[str]) -> str:
    return " ".join(hand)

# Hand evaluation
# Returns a tuple (rank_class, tie_breakers...) higher is better
# rank_class order:
# 8: Straight Flush
# 7: Four of a Kind
# 6: Full House
# 5: Flush
# 4: Straight
# 3: Three of a Kind
# 2: Two Pair
# 1: One Pair
# 0: High Card
def evaluate_hand(hand: List[str]) -> Tuple:
    ranks = sorted([card_rank(c) for c in hand], reverse=True)
    suits = [card_suit(c) for c in hand]
    rank_counts: Dict[int, int] = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    # Handle wheel straight (A-2-3-4-5)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_flush = len(set(suits)) == 1
    is_straight, straight_high = _is_straight(unique_ranks)

    # Group ranks by counts (for sorting ties)
    # e.g., Four of a Kind -> [(4, rank), (1, kicker)]
    groups = sorted(((cnt, r) for r, cnt in rank_counts.items()), key=lambda x: (x[0], x[1]), reverse=True)
    counts_sorted = [cnt for cnt, _ in groups]
    ranks_by_group_then_rank = [r for _, r in groups]
    kickers_desc = sorted((r for r in ranks if rank_counts[r] == 1), reverse=True)

    if is_straight and is_flush:
        return (8, straight_high)
    if counts_sorted[0] == 4:
        # Four of a kind: (7, quad_rank, kicker)
        quad_rank = groups[0][1]
        kicker = max(r for r in ranks if r != quad_rank)
        return (7, quad_rank, kicker)
    if counts_sorted[0] == 3 and counts_sorted[1] == 2:
        # Full house: (6, trips_rank, pair_rank)
        trips_rank = groups[0][1]
        pair_rank = groups[1][1]
        return (6, trips_rank, pair_rank)
    if is_flush:
        return (5, *sorted(ranks, reverse=True))
    if is_straight:
        return (4, straight_high)
    if counts_sorted[0] == 3:
        # Trips: (3, trips_rank, kickers...)
        trips_rank = groups[0][1]
        other = sorted((r for r in ranks if r != trips_rank), reverse=True)
        return (3, trips_rank, *other)
    if counts_sorted[0] == 2 and counts_sorted[1] == 2:
        # Two pair: (2, high_pair, low_pair, kicker)
        pair1, pair2 = sorted((groups[0][1], groups[1][1]), reverse=True)
        kicker = max(r for r in ranks if r != pair1 and r != pair2)
        return (2, pair1, pair2, kicker)
    if counts_sorted[0] == 2:
        # One pair: (1, pair_rank, kickers...)
        pair_rank = groups[0][1]
        other = sorted((r for r in ranks if r != pair_rank), reverse=True)
        return (1, pair_rank, *other)
    # High card
    return (0, *sorted(ranks, reverse=True))


def _is_straight(unique_desc_ranks: List[int]) -> Tuple[bool, int]:
    # unique ranks sorted descending
    if len(unique_desc_ranks) < 5:
        return False, 0
    # Check normal straight
    best_high = None
    for i in range(len(unique_desc_ranks) - 4):
        window = unique_desc_ranks[i:i + 5]
        if window[0] - window[4] == 4 and len(window) == 5:
            best_high = window[0]
            break
    # Check wheel A-2-3-4-5
    if not best_high and set([14, 5, 4, 3, 2]).issubset(set(unique_desc_ranks)):
        return True, 5
    return (best_high is not None), (best_high or 0)


def compare_hands(h1: List[str], h2: List[str]) -> int:
    e1 = evaluate_hand(h1)
    e2 = evaluate_hand(h2)
    if e1 > e2:
        return 1
    if e2 > e1:
        return -1
    return 0


# Opponent AI (very simple)
def ai_choose_discards(hand: List[str]) -> List[int]:
    # Return 0-based indices to discard
    ranks = [card_rank(c) for c in hand]
    counts: Dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1

    eval_rank = evaluate_hand(hand)[0]

    # Keep made hands (two pair or better), discard nothing
    if eval_rank >= 2:
        return []

    # If one pair: keep the pair, discard others
    if eval_rank == 1:
        pair_rank = _find_pair_rank(counts)
        return [i for i, r in enumerate(ranks) if r != pair_rank]

    # No pair: try to keep high cards (A, K, Q) up to 2, discard others
    keep_idxs = [i for i, r in enumerate(ranks) if r >= RANK_VALUE["Q"]]
    if len(keep_idxs) > 2:
        keep_idxs = keep_idxs[:2]
    discard = [i for i in range(len(hand)) if i not in keep_idxs]
    # never discard all 5, cap at 3 for basic draw strategy
    if len(discard) > 3:
        discard = discard[:3]
    return discard


def _find_pair_rank(counts: Dict[int, int]) -> int:
    for r, c in counts.items():
        if c == 2:
            return r
    return -1


def ai_bet_decision(stage: str, hand: List[str], can_raise: bool, to_call: int, ai_bank: int, pot: int) -> Tuple[str, int]:
    # stage: "pre" or "post"
    # Returns (action, amount). action in {"fold","check","call","bet","raise"}
    strength = evaluate_hand(hand)[0]
    # crude aggression based on strength and stage
    if to_call > 0:
        # facing a bet
        if strength >= 3:  # Trips or better: continue, raise if possible
            if can_raise and ai_bank > to_call and pot >= 2:
                raise_amt = min(ai_bank - to_call, max(2, pot // 4))
                return "raise", max(1, raise_amt)
            return "call", to_call
        if strength == 2:  # two pair
            # call modest bets, fold to big ones
            if to_call <= max(2, pot // 3) or to_call <= ai_bank // 6:
                return "call", to_call
            return "fold", 0
        if strength == 1:  # one pair
            if to_call <= max(1, pot // 5):
                return "call", to_call
            return "fold", 0
        # high card
        if to_call <= 1 and random.random() < 0.3:
            return "call", to_call
        return "fold", 0
    else:
        # no bet yet
        if strength >= 3 and ai_bank > 0:
            bet_size = min(ai_bank, max(2, pot // 4 if pot > 0 else 2))
            return "bet", bet_size
        if stage == "post" and strength == 2:
            return "bet", min(ai_bank, max(2, pot // 4 if pot > 0 else 2))
        # otherwise check
        return "check", 0


# Game flow
class GameState:
    def __init__(self):
        self.player_bank = 50
        self.ai_bank = 50
        self.pot = 10
        self.deck: List[str] = []
        self.player_hand: List[str] = []
        self.ai_hand: List[str] = []
        self.round_num = 1

    def new_hand(self):
        self.pot = 10
        self.player_bank -= 5
        self.ai_bank -= 5
        self.deck = make_deck()
        shuffle_deck(self.deck)
        self.player_hand = deal(self.deck, 5)
        self.ai_hand = deal(self.deck, 5)


def get_positive_int(prompt: str, max_value: int = None) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw.isdigit():
            print("Please enter a positive whole number.")
            continue
        val = int(raw)
        if val <= 0:
            print("Number must be above 0.")
            continue
        if max_value is not None and val > max_value:
            print(f"Maximum allowed is {max_value}.")
            continue
        return val


def get_discard_indices(num_cards: int) -> List[int]:
    print("Enter space-separated card indexes to discard (1 is the first index). Enter 0 to keep all.")
    while True:
        raw = input("Discard indexes: ").strip()
        if raw == "":
            print("Enter 0 to keep all or specify indexes.")
            continue
        parts = raw.split()
        if len(parts) == 1 and parts[0] == "0":
            return []
        try:
            idxs = [int(p) for p in parts]
        except ValueError:
            print("Please enter numbers only.")
            continue
        # Validate: all > 0 and within range
        if any(i <= 0 for i in idxs):
            print("All card indexes must be above 0.")
            continue
        if any(i > num_cards for i in idxs):
            print(f"Card indexes must be between 1 and {num_cards}.")
            continue
        # Deduplicate while preserving order
        seen = set()
        clean = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                clean.append(i)
        if len(clean) > 3:
            print("You can discard at most 3 cards.")
            continue
        return [i - 1 for i in clean]  # convert to 0-based


def betting_round(state: GameState, stage: str) -> bool:
    # Returns False if someone folded and hand should end
    # At most one raise per round
    to_call = 0
    last_bettor = None
    has_raised = False

    def player_turn():
        nonlocal to_call, last_bettor, has_raised
        print(f"Your hand: {hand_to_str(state.player_hand)}")
        print(f"Bankrolls - You: ${state.player_bank} | Opponent: ${state.ai_bank} | Pot: ${state.pot}")
        if to_call > 0:
            # options: call, raise (if not raised), fold
            print(f"Opponent bet ${to_call}. Options: [1] Call, [2] Raise, [3] Fold")
            choice = input("Choose action (1/2/3): ").strip()
            if choice == "1":
                # call
                call_amt = min(state.player_bank, to_call)
                state.player_bank -= call_amt
                state.pot += call_amt
                to_call -= call_amt
                last_bettor = "ai"
                print(f"You call ${call_amt}.")
            elif choice == "2" and not has_raised:
                max_raise = state.player_bank
                if max_raise <= 0:
                    print("You cannot raise (no chips). Defaulting to call if possible.")
                    return player_turn()
                raise_amt = get_positive_int(f"Enter raise amount (1 to {max_raise}): ", max_raise)
                # pay call + raise
                total = min(state.player_bank, to_call + raise_amt)
                state.player_bank -= total
                state.pot += total
                to_call = raise_amt  # amount for opponent to call
                has_raised = True
                last_bettor = "player"
                print(f"You raise to ${raise_amt}.")
            elif choice == "3":
                print("You fold.")
                # Opponent wins pot
                state.ai_bank += state.pot
                state.pot = 0
                return False
            else:
                print("Invalid choice.")
                return player_turn()
        else:
            # options: check or bet
            print("Options: [1] Check, [2] Bet")
            choice = input("Choose action (1/2): ").strip()
            if choice == "1":
                print("You check.")
                last_bettor = "player"
            elif choice == "2":
                max_bet = state.player_bank
                if max_bet <= 0:
                    print("You cannot bet (no chips). Checking instead.")
                    print("You check.")
                    last_bettor = "player"
                else:
                    bet_amt = get_positive_int(f"Enter bet amount (1 to {max_bet}): ", max_bet)
                    state.player_bank -= bet_amt
                    state.pot += bet_amt
                    to_call = bet_amt
                    last_bettor = "player"
                    print(f"You bet ${bet_amt}.")
            else:
                print("Invalid choice.")
                return player_turn()
        return True

    def ai_turn():
        nonlocal to_call, last_bettor, has_raised
        action, amount = ai_bet_decision(stage, state.ai_hand, can_raise=(not has_raised), to_call=to_call,
                                         ai_bank=state.ai_bank, pot=state.pot)
        if to_call > 0:
            if action == "fold":
                print("Opponent folds.")
                state.player_bank += state.pot
                state.pot = 0
                return False
            if action == "call":
                pay = min(state.ai_bank, to_call)
                state.ai_bank -= pay
                state.pot += pay
                to_call -= pay
                last_bettor = "player"
                print(f"Opponent calls ${pay}.")
            elif action == "raise" and not has_raised and state.ai_bank > to_call:
                raise_amt = min(state.ai_bank - to_call, max(1, amount))
                total = to_call + raise_amt
                pay = min(state.ai_bank, total)
                state.ai_bank -= pay
                state.pot += pay
                to_call = raise_amt
                has_raised = True
                last_bettor = "ai"
                print(f"Opponent raises to ${raise_amt}.")
            else:
                # default to call
                pay = min(state.ai_bank, to_call)
                state.ai_bank -= pay
                state.pot += pay
                to_call -= pay
                last_bettor = "player"
                print(f"Opponent calls ${pay}.")
        else:
            if action == "bet" and state.ai_bank > 0:
                bet_amt = min(state.ai_bank, max(1, amount))
                state.ai_bank -= bet_amt
                state.pot += bet_amt
                to_call = bet_amt
                last_bettor = "ai"
                print(f"Opponent bets ${bet_amt}.")
            else:
                print("Opponent checks.")
                last_bettor = "ai"
        return True

    # Simple order: player acts first pre-draw, post-draw alternate based on round num parity
    player_first = True if stage == "pre" else False

    while True:
        if player_first:
            if not player_turn():
                return False
            if to_call == 0 and last_bettor == "player":
                # if both checked -> end, or after call -> end
                # Now AI acts
                if not ai_turn():
                    return False
                if to_call == 0:
                    break
                # go back to player to resolve call
                continue
            else:
                # opponent to respond
                if not ai_turn():
                    return False
                if to_call == 0:
                    break
                # player needs to respond possibly to raise
                player_first = True  # stay in loop
                continue
        else:
            if not ai_turn():
                return False
            if to_call == 0 and last_bettor == "ai":
                if not player_turn():
                    return False
                if to_call == 0:
                    break
                continue
            else:
                if not player_turn():
                    return False
                if to_call == 0:
                    break
                continue

    return True


def draw_phase(state: GameState):
    # Player discards
    print(f"Your hand: {hand_to_str(state.player_hand)}")
    discard_idxs = get_discard_indices(len(state.player_hand))
    discard_idxs = sorted(discard_idxs, reverse=True)
    for i in discard_idxs:
        state.player_hand.pop(i)
    new_cards = deal(state.deck, 5 - len(state.player_hand))
    state.player_hand.extend(new_cards)
    print(f"You draw {len(new_cards)}: {hand_to_str(state.player_hand)}")

    # AI discards
    ai_discard = ai_choose_discards(state.ai_hand)
    ai_discard = sorted(ai_discard, reverse=True)
    for i in ai_discard:
        state.ai_hand.pop(i)
    ai_new = deal(state.deck, 5 - len(state.ai_hand))
    state.ai_hand.extend(ai_new)
    print(f"Opponent draws {len(ai_new)} cards.")


def showdown(state: GameState):
    print(f"Your hand: {hand_to_str(state.player_hand)}")
    print(f"Opponent's hand: {hand_to_str(state.ai_hand)}")
    res = compare_hands(state.player_hand, state.ai_hand)
    if res > 0:
        print("You win the pot!")
        state.player_bank += state.pot
    elif res < 0:
        print("Opponent wins the pot.")
        state.ai_bank += state.pot
    else:
        print("It's a tie. Pot is split.")
        half = state.pot // 2
        state.player_bank += half
        state.ai_bank += state.pot - half
    state.pot = 0


def play_hand(state: GameState) -> bool:
    print("\n" + "=" * 50)
    print(f"Hand #{state.round_num}")
    state.new_hand()

    # Pre-draw betting
    if not betting_round(state, stage="pre"):
        return True  # someone folded; continue game if bankrolls remain

    # Draw
    draw_phase(state)

    # Post-draw betting
    if not betting_round(state, stage="post"):
        return True

    # Showdown
    showdown(state)
    state.round_num += 1
    return True


def main():
    print("Welcome to 5-Card Draw Poker!")
    print("You and the opponent each start with $50.")
    print("Minimum bet is $5.")
    state = GameState()

    while state.player_bank > 0 and state.ai_bank > 0:
        keep_playing = play_hand(state)
        if not keep_playing:
            break
        if state.player_bank <= 0 or state.ai_bank <= 0:
            break
        # Ask to continue
        ans = input("Play another hand? (Enter or y/n): ").strip().lower()
        if ans not in ("y", "yes", ""):
            break

    print("\nFinal bankrolls:")
    print(f"You: ${state.player_bank} | Opponent: ${state.ai_bank}")
    if state.player_bank > state.ai_bank:
        print("You come out ahead. Well played!")
    elif state.player_bank < state.ai_bank:
        print("Opponent has the edge this time.")
    else:
        print("It's a wash. Thanks for playing!")


if __name__ == "__main__":
    main()
