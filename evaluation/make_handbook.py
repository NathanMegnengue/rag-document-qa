"""Generate evaluation/handbook.pdf: a fictional employee handbook used to evaluate the API.

Everything in it is invented (company, numbers, contacts), so it is free of any copyright and the
correct answers are known for certain. Regenerate with:

    pip install reportlab
    python -m evaluation.make_handbook
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent / "handbook.pdf"

# One section per page. The page numbers are referenced by questions.json.
PAGES: list[tuple[str, list[str]]] = [
    (
        "1. About Bluefern Robotics",
        [
            "Bluefern Robotics is a fictional company, invented for testing purposes. "
            "It designs and builds robots that pick and sort parcels in warehouses.",
            "The company was founded in 2011 and its headquarters are in Lyon. "
            "A second office is located in Montreal. Today Bluefern employs 240 people.",
            "Our mission is to make warehouse work safer and more reliable. "
            "Every new employee receives a copy of this handbook on the first day.",
            "Bluefern's robots are used by logistics companies across Europe and North America. The engineering teams work in small cross-functional squads, each responsible for one product area."
            "We value clear communication, curiosity and respect for colleagues, customers and the people who work next to our machines.",
        ],
    ),
    (
        "2. Working hours and remote work",
        [
            "The standard working week is 38 hours. Core hours, during which everyone "
            "is expected to be reachable, are from 10:00 to 16:00.",
            "After the three-month probation period, employees may work remotely up to "
            "3 days per week, with the agreement of their manager.",
            "In July and August, Fridays end earlier: the working day finishes at 15:00.",
            "Managers and employees agree on a weekly schedule together. Meetings are kept short and are planned inside the core hours whenever possible."
            "Overtime must be approved in advance by the manager and is compensated with additional time off rather than with extra pay.",
        ],
    ),
    (
        "3. Leave and absences",
        [
            "Every employee receives 27 days of paid annual leave per year. After 5 years "
            "of service, 5 extra days are added.",
            "A medical certificate is required for sick leave after 3 consecutive days of "
            "absence. The second parent is entitled to 16 weeks of parental leave.",
            "Unused leave can be carried over to the next year, up to a maximum of 5 days, "
            "and must be taken before March 31.",
            "Leave requests are submitted through the HR portal and approved by the manager. Requests for the summer are best submitted early so that every team stays covered."
            "An employee who is ill should inform the manager as early as possible on the first day of absence.",
        ],
    ),
    (
        "4. Business travel and expenses",
        [
            "On business trips, the daily meal allowance is 18 euros. Hotel stays are "
            "reimbursed up to 140 euros per night in Europe and 190 euros per night elsewhere.",
            "The train is the preferred means of transport for trips shorter than 4 hours.",
            "Expense reports must be submitted within 30 days after the trip. A receipt is "
            "required for every expense above 25 euros.",
            "Taxis are reimbursed only when no public transport is available. Flights are booked through the company travel desk, in economy class."
            "Personal expenses, such as minibar items or entertainment, are never reimbursed.",
        ],
    ),
    (
        "5. IT security",
        [
            "Passwords must contain at least 14 characters, and multi-factor authentication "
            "is mandatory on all company accounts.",
            "A phishing email must be reported within 1 hour to security@bluefern.example. "
            "Laptops are encrypted, and USB drives of unknown origin must never be plugged in.",
            "When working on public Wi-Fi, the company VPN must always be switched on.",
            "Company data must be stored only on approved systems. Sharing confidential files through personal email or private messaging apps is not allowed."
            "Lost or stolen devices must be reported to the IT helpdesk immediately so that they can be locked remotely.",
        ],
    ),
    (
        "6. Learning and development",
        [
            "Each employee has an annual learning budget of 1,500 euros. Attending a "
            "conference requires the approval of the manager.",
            "Every week, 2 hours are reserved for Friday Labs, a time to explore new ideas. "
            "The mentorship programme lasts 6 months, and mentors apply each January.",
            "The learning budget can be used for courses, books and certifications related to the employee's work. Unused budget does not carry over to the next year."
            "Employees are encouraged to share what they learn through short internal talks.",
        ],
    ),
    (
        "7. Code of conduct",
        [
            "Concerns can be reported to the ethics line at 0800 555 0142 or by email to "
            "ethics@bluefern.example. Nobody may suffer retaliation for reporting a concern in "
            "good faith.",
            "Gifts worth more than 50 euros must be declared. Any conflict of interest must be "
            "declared to the HR department within 10 days.",
            "The company expects honest and respectful behaviour from everyone. Harassment and discrimination of any kind are not tolerated."
            "Confidential information about customers must never be shared outside the company.",
        ],
    ),
    (
        "8. Equipment and leaving the company",
        [
            "Laptops are replaced every 3 years. When leaving the company, the laptop must be "
            "returned within 5 working days after the last day of work.",
            "The probation period lasts 3 months. After it, the notice period for engineers "
            "is 2 months.",
            "Equipment remains the property of the company. Employees are responsible for taking reasonable care of it and for reporting any damage."
            "On the last day, access to company systems is closed and a short exit interview is offered.",
        ],
    ),
]


def build(path: Path = OUT) -> Path:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=11, leading=16)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        title="Bluefern Robotics - Employee Handbook (fictional)",
        author="Fictional test document",
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
    )
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 0.4 * cm))
        for text in paragraphs:
            story.append(Paragraph(text, body))
            story.append(Spacer(1, 0.3 * cm))
    doc.build(story)
    return path


if __name__ == "__main__":
    print(f"Wrote {build()}")
