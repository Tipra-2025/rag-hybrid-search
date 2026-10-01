# IT Security Policy

## Purpose

This IT Security Policy establishes mandatory security standards for all Acme Corporation employees, contractors, and third parties who access company systems or data. Non-compliance may result in disciplinary action, termination, or legal liability.

## Scope

This policy applies to:
- All company-owned and personal devices used to access company systems.
- All cloud services, SaaS applications, and on-premise systems used for company business.
- All employees, contractors, and third-party vendors.

## Password Policy

All passwords must:
- Be at least **14 characters** long.
- Contain uppercase letters, lowercase letters, numbers, and at least one special character.
- Not be reused across different systems or within the last 12 passwords.
- Be changed every **90 days** for privileged accounts and every **180 days** for standard accounts.

**Password sharing is strictly prohibited.** Never write passwords down, send them via email, or store them in unencrypted files.

Use the company-approved password manager (1Password Enterprise) for storing credentials.

## Multi-Factor Authentication (MFA)

MFA is mandatory for:
- All access to company email and collaboration tools (Microsoft 365).
- All VPN connections.
- All cloud management consoles (AWS, Azure, GCP).
- All HR and finance systems.
- Any privileged or administrative access.

The approved MFA methods are: hardware security keys (FIDO2), authenticator apps (Microsoft Authenticator). SMS-based MFA is not permitted for privileged access.

## Device Security

All devices used for company work must:
- Have full-disk encryption enabled (BitLocker on Windows, FileVault on macOS).
- Have the latest operating system security patches installed within 14 days of release.
- Have approved endpoint protection (CrowdStrike Falcon) installed and active.
- Be enrolled in the company's Mobile Device Management (MDM) system.
- Lock automatically after 5 minutes of inactivity.
- Require authentication to unlock.

Personal devices (BYOD) may only access company systems through the company's VDI solution or approved mobile apps, not through native access to company data.

## Data Classification

Acme data is classified into four levels:

| Level | Description | Examples |
|---|---|---|
| **Public** | May be shared freely | Marketing materials, published reports |
| **Internal** | For employees only | Internal announcements, process docs |
| **Confidential** | Business-sensitive | Customer data, financial records, contracts |
| **Restricted** | Highest sensitivity | PII, trade secrets, source code, keys |

Restricted and Confidential data must be encrypted in transit and at rest. Access is on a need-to-know basis, reviewed quarterly.

## Acceptable Use

Company systems may only be used for legitimate business purposes. The following are prohibited:

- Downloading or distributing copyrighted content without a license.
- Accessing or storing adult content, gambling sites, or content promoting violence.
- Installing unauthorized software or browser extensions.
- Using company email for personal commercial activity.
- Sending Confidential or Restricted data to personal email accounts or personal cloud storage.
- Cryptocurrency mining or any activity that consumes disproportionate compute resources.
- Connecting to unauthorized or untrusted Wi-Fi networks for company work without VPN.

## Remote Access

When working outside company premises:
- Always connect via company VPN before accessing internal resources.
- Do not conduct sensitive calls in public places where you may be overheard.
- Physically secure your device — never leave it unattended in a visible location.
- Use privacy screens when working in public.

## Incident Reporting

Report all suspected security incidents immediately to security@acme.com or call the security hotline at +1-800-ACME-SEC.

Incidents include:
- Lost or stolen devices.
- Phishing emails (report via the "Report Phishing" button in Outlook before deleting).
- Suspected malware or ransomware.
- Unauthorized access to systems.
- Accidental disclosure of Confidential or Restricted data.

**Do not attempt to investigate or remediate a security incident yourself.** Contact IT Security immediately.

## Social Engineering and Phishing

Be alert to attempts to trick you into revealing credentials or sensitive information:
- Acme IT will **never** ask for your password.
- Verify unexpected requests for wire transfers or sensitive data with the requester via a known phone number.
- If you receive a suspicious email, do not click links — report it immediately.

## Third-Party and Vendor Access

Vendors with access to company systems must:
- Complete a security assessment prior to access being granted.
- Sign the Vendor Security Agreement.
- Access only the systems and data necessary for their work (principle of least privilege).
- Notify IT Security of any security incidents involving company data within 24 hours.

## Policy Violations

Violations will be reviewed by IT Security and HR. Consequences range from mandatory retraining to termination, depending on severity and intent. Legal action may be taken where criminal activity is involved.

## Policy Review

This policy is reviewed annually by the Chief Information Security Officer. Employees are notified of material changes. The current version is always available on the company intranet.

Last reviewed: October 2026  
Next review: October 2027  
Owner: Chief Information Security Officer
