"""
Expense Splitter with Debt Minimization
========================================
Tracks shared expenses among a group and computes the
minimum number of transactions needed to settle all debts
using a greedy algorithm on net balances.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Expense:
    """Represents a single shared expense."""
    description: str
    amount: float
    paid_by: str
    shared_by: List[str]

    @property
    def share(self) -> float:
        """Per-person share of this expense."""
        return self.amount / len(self.shared_by)


@dataclass
class Group:
    """Holds all members and expenses for a shared-expense group."""
    name: str
    members: List[str] = field(default_factory=list)
    expenses: List[Expense] = field(default_factory=list)

    def add_member(self, member: str) -> None:
        """Add a unique member to the group."""
        if member in self.members:
            raise ValueError(f"Member '{member}' already exists.")
        self.members.append(member)

    def add_expense(self, expense: Expense) -> None:
        """Validate and record an expense."""
        unknown = set([expense.paid_by] + expense.shared_by) - set(self.members)
        if unknown:
            raise ValueError(f"Unknown member(s): {', '.join(unknown)}")
        if expense.amount <= 0:
            raise ValueError("Expense amount must be positive.")
        self.expenses.append(expense)


# ---------------------------------------------------------------------------
# Core Algorithm: Net Balance Computation
# ---------------------------------------------------------------------------

def compute_net_balances(group: Group) -> Dict[str, float]:
    """
    Compute each member's net balance across all expenses.

    A positive balance means the member is owed money;
    a negative balance means the member owes money.

    Time complexity: O(E * S) where E = expenses, S = avg sharers per expense.
    """
    balances: Dict[str, float] = defaultdict(float)

    for expense in group.expenses:
        balances[expense.paid_by] += expense.amount
        for person in expense.shared_by:
            balances[person] -= expense.share

    return dict(balances)


# ---------------------------------------------------------------------------
# Core Algorithm: Debt Minimization (Greedy)
# ---------------------------------------------------------------------------

def minimize_transactions(balances: Dict[str, float]) -> List[Tuple[str, str, float]]:
    """
    Compute the minimum number of transactions to settle all debts.

    Algorithm:
        1. Separate members into creditors (positive balance) and
           debtors (negative balance).
        2. Greedily match the largest creditor with the largest debtor.
        3. Settle as much as possible in each step, carrying over
           any remainder.

    This greedy approach yields an optimal solution when balances
    can be matched freely (no currency constraints).

    Time complexity: O(N log N) due to sorting, O(N) settlements.

    Args:
        balances: Net balance per member (positive = owed, negative = owes).

    Returns:
        List of (payer, receiver, amount) settlement tuples.
    """
    # Filter out near-zero balances to avoid floating-point noise
    creditors = sorted(
        ((name, bal) for name, bal in balances.items() if bal > 1e-9),
        key=lambda x: -x[1],
    )
    debtors = sorted(
        ((name, -bal) for name, bal in balances.items() if bal < -1e-9),
        key=lambda x: -x[1],
    )

    settlements: List[Tuple[str, str, float]] = []
    ci, di = 0, 0  # creditor index, debtor index

    while ci < len(creditors) and di < len(debtors):
        creditor, credit = creditors[ci]
        debtor, debt = debtors[di]

        settled = min(credit, debt)
        settlements.append((debtor, creditor, round(settled, 2)))

        credit -= settled
        debt -= settled

        # Advance whichever side is fully settled
        if credit < 1e-9:
            ci += 1
        else:
            creditors[ci] = (creditor, credit)

        if debt < 1e-9:
            di += 1
        else:
            debtors[di] = (debtor, debt)

    return settlements


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(group: Group) -> str:
    """
    Build a human-readable settlement report for the group.

    Returns:
        Formatted multi-line string with balances and settlement steps.
    """
    balances = compute_net_balances(group)
    settlements = minimize_transactions(balances)

    lines = [f"\n=== {group.name} — Expense Report ===\n"]

    lines.append("Net Balances:")
    for member, bal in sorted(balances.items(), key=lambda x: -x[1]):
        status = "gets back" if bal >= 0 else "owes"
        lines.append(f"  {member:<15} {status}  ₹{abs(bal):.2f}")

    lines.append(f"\nMinimum Settlements ({len(settlements)} transaction(s)):")
    if settlements:
        for payer, receiver, amount in settlements:
            lines.append(f"  {payer} → {receiver}  ₹{amount:.2f}")
    else:
        lines.append("  All settled — no transactions needed.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo / Entry Point
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Demonstrate the expense splitter with a sample scenario."""
    group = Group(name="Goa Trip")

    for member in ["Alice", "Bob", "Carol", "Dave"]:
        group.add_member(member)

    group.add_expense(Expense("Hotel", 4000, "Alice", ["Alice", "Bob", "Carol", "Dave"]))
    group.add_expense(Expense("Dinner", 1200, "Bob",   ["Alice", "Bob", "Carol"]))
    group.add_expense(Expense("Taxi",    600, "Carol",  ["Bob", "Carol", "Dave"]))
    group.add_expense(Expense("Drinks",  800, "Dave",   ["Alice", "Dave"]))

    print(generate_report(group))


if __name__ == "__main__":
    run_demo()
