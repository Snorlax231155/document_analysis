import pymupdf as fitz
from pathlib import Path

def create_enterprise_test_pdf(output_path: str):
    doc = fitz.open()

    # Page 1: HR Policy (HR-204)
    page1 = doc.new_page()
    p1_text = """ACME Global Corporation — Enterprise Policy Handbook (2026 Edition)

Section 1: Human Resources & Annual Leave Entitlements
Policy Reference Code: HR-204
Effective Date: January 1, 2026
Applicability: All Full-Time and Permanent Employees Global

1.1 Paid Annual Leave Entitlement
Full-time employees at ACME Global Corporation are entitled to twenty (20) business days of paid annual leave per calendar year. Annual leave accrues on a monthly pro-rata basis starting from the employee's first full calendar month of employment.

1.2 Leave Application & Approval Notice
All planned annual leave requests exceeding three (3) consecutive business days must be formally submitted through the HR Portal at least fourteen (14) calendar days prior to the proposed start date.

1.3 Carry-Over & Rollover Limits
A maximum of five (5) unused annual leave days may be carried forward into the subsequent calendar year. Any carried-over leave must be utilized within the first quarter (Q1, by March 31). Unused leave beyond five days is forfeited without monetary cash-out unless required by local labor law.
"""
    page1.insert_text((50, 50), p1_text, fontsize=11)

    # Page 2: Information Security (SEC-402)
    page2 = doc.new_page()
    p2_text = """ACME Global Corporation — Enterprise Policy Handbook (2026 Edition)

Section 2: Information Security & Remote Work Infrastructure
Policy Reference Code: SEC-402
Classification: STRICTLY CONFIDENTIAL / INTERNAL USE ONLY
Compliance Contact: security-compliance@acme-global.com

2.1 Remote Access & VPN Mandatory Standards
All employees accessing ACME production databases, internal code repositories, or customer data remotely must connect exclusively through the ACME Enterprise Zero-Trust VPN gateway. Usage of unencrypted public Wi-Fi networks (e.g., airport or coffee shop networks) without an active VPN tunnel is strictly prohibited under SEC-402.

2.2 Authentication & Hardware Security Keys
Multi-Factor Authentication (MFA) is mandatory across all enterprise accounts. Access to administrative web consoles, AWS/GCP cloud environments, and code deployment pipelines requires a registered FIDO2 hardware security key (e.g., YubiKey 5 Series).

2.3 Workstation Disk Encryption & Endpoint Monitoring
All company-issued laptops must run FileVault (macOS) or BitLocker (Windows 11 Enterprise) with 256-bit AES encryption. Disk decryption keys are escrowed with the ACME Security Operations Center (SOC).
"""
    page2.insert_text((50, 50), p2_text, fontsize=11)

    # Page 3: Financial Expense & Travel (FIN-901)
    page3 = doc.new_page()
    p3_text = """ACME Global Corporation — Enterprise Policy Handbook (2026 Edition)

Section 3: Corporate Travel & Expense Reimbursement Regulations
Policy Reference Code: FIN-901
Department: Global Finance & Accounting
Reimbursement Processing Window: 30 Calendar Days

3.1 Per-Diem Meal & Lodging Allowances
The maximum daily per-diem allowance for meals during domestic corporate travel is $150 USD per day. Lodging expenses must be booked via the Corporate Travel Portal (Concur) with a nightly cap of $280 USD for standard business hotels.

3.2 Commercial Flight Booking Rules
Air travel for domestic and short-haul international flights (under 6 hours duration) must be booked in Economy Standard class. Business class travel is permissible only for long-haul flights exceeding eight (8) hours of non-stop travel time with prior Vice President written approval.

3.3 Expense Submission Timelines
All itemized receipts and corporate credit card expense reports must be submitted into the Finance Portal within thirty (30) calendar days following the completion of travel. Late submissions past 60 days will require Chief Financial Officer (CFO) sign-off.
"""
    page3.insert_text((50, 50), p3_text, fontsize=11)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_file))
    doc.close()
    print(f"Created enterprise test PDF at: {output_file.resolve()}")

if __name__ == "__main__":
    create_enterprise_test_pdf("data/test_docs/acme_enterprise_security_and_hr_policy_2026.pdf")
