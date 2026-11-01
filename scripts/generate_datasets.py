"""Generate fully synthetic cybersecurity telemetry; deterministic, no external data."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets'
SEED = 20261031


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def missing(rng, frame, cols, rate):
    for col in cols:
        frame.loc[rng.random(len(frame)) < rate, col] = np.nan
    return frame


def phishing(rng, n=6000):
    y = (rng.random(n) < 0.17).astype(int)
    kind = np.array(['ordinary'] * n, dtype=object)
    kind[(y == 0) & (rng.random(n) < .20)] = 'marketing'
    kind[(y == 0) & (rng.random(n) < .06)] = 'admin_notice'
    kind[(y == 1) & (rng.random(n) < .31)] = 'stealth_phish'
    p = y == 1
    marketer = kind == 'marketing'
    stealth = kind == 'stealth_phish'
    rep = np.clip(rng.normal(82, 15, n) - p * 35 + stealth * 26 - marketer * 10, 1, 100).round(1)
    url_count = rng.poisson(1.0 + p * 1.4 + marketer * 2.4, n)
    suspicious = rng.binomial(url_count, np.clip(.035 + p * .31 - stealth * .20 + marketer * .018, 0, .9))
    urgency = np.clip(rng.beta(2, 5, n) + p * .38 - stealth * .27 + (kind == 'admin_notice') * .42 + marketer * .22, 0, 1).round(3)
    short = (rng.random(n) < np.clip(.035 + p * .30 - stealth * .19 + marketer * .25, 0, .92)).astype(int)
    attach = rng.poisson(.18 + .38 * p + .14 * marketer, n)
    spf_pass = rng.random(n) < np.clip(.92 - p * .48 + stealth * .36 + marketer * .045, 0, 1)
    dkim_pass = rng.random(n) < np.clip(.93 - p * .45 + stealth * .35, 0, 1)
    dmarc_pass = rng.random(n) < np.clip(.94 - p * .49 + stealth * .37, 0, 1)
    spf = np.where(spf_pass, 'pass', rng.choice(['fail', 'softfail', 'none'], n, p=[.55,.25,.20]))
    dkim = np.where(dkim_pass, 'pass', rng.choice(['fail', 'none'], n, p=[.75,.25]))
    dmarc = np.where(dmarc_pass, 'pass', rng.choice(['fail', 'none'], n, p=[.8,.2]))
    domains = np.array(['corp-notify.example', 'secure-mail.example', 'hr-updates.example', 'updates.example',
                        'service-desk.example', 'mailer.example', 'rewards.example', 'offers.example',
                        'ticketing.example', 'support-portal.example', 'notify-gateway.example', 'invoices.example'])
    # .example is reserved. No active malicious domains or URLs.
    domain_idx = rng.integers(0, len(domains), size=n)
    domain_idx[p & ~stealth] = rng.choice([2, 4, 9, 10, 11], size=np.sum(p & ~stealth))
    legitimate_subject = ['Your weekly account summary', 'Team meeting notes', 'Project update', 'IT service notification',
                          'Subscription renewal reminder', 'System change advisory']
    phish_subject = ['Urgent: verify your account', 'Action required: payment declined',
                     'Password expires today', 'Document shared with you', 'Review security alert']
    marketing_subject = ['Limited-time member offer', 'Your monthly newsletter', 'Exclusive rewards update']
    notices_subject = ['Immediate scheduled maintenance', 'Required security training reminder']
    subjects = []
    bodies = []
    for i in range(n):
        if kind[i] == 'marketing':
            sub = marketing_subject[int(rng.integers(len(marketing_subject)))]
            body = 'This promotional message contains links to view your offers and manage preferences. Please review the campaign before the deadline.'
        elif kind[i] == 'admin_notice':
            sub = notices_subject[int(rng.integers(len(notices_subject)))]
            body = 'Please complete the approved internal security update today. Contact the service desk if you need assistance.'
        elif y[i]:
            sub = phish_subject[int(rng.integers(len(phish_subject)))] if not stealth[i] else 'Shared file awaiting review'
            body = ('Confirm your sign-in to prevent an interruption. Open the supplied verification link and complete the requested steps.'
                    if not stealth[i] else 'A document was shared with your team. Review the file when convenient. The link is available in the notification.')
        else:
            sub = legitimate_subject[int(rng.integers(len(legitimate_subject)))]
            body = 'Here is your routine work update. Follow the documented process or contact your support team for more information.'
        subjects.append(sub)
        bodies.append(body + (' Reference: CASE-%04d.' % (i % 1000)))
    subject_length = np.array([len(s) for s in subjects]) + rng.integers(-2, 5, n)
    body_length = np.array([len(s) for s in bodies]) + rng.integers(0, 280, n)
    frame = pd.DataFrame({
        'email_id': [f'MAIL-{i:06d}' for i in range(1, n+1)], 'sender_domain': domains[domain_idx],
        'sender_reputation': rep, 'spf_result': spf, 'dkim_result': dkim, 'dmarc_result': dmarc,
        'url_count': url_count, 'suspicious_url_count': suspicious, 'attachment_count': attach,
        'has_shortened_url': short, 'urgent_language_score': urgency, 'body_length': body_length,
        'subject_length': subject_length, 'subject_text': subjects, 'body_text': bodies,
        'message_theme': rng.choice(['blue', 'green', 'amber', 'grey'], n),
        'phishing_label': y})
    return missing(rng, frame, ['sender_reputation', 'spf_result', 'body_length', 'dmarc_result'], .045)


def edr(rng, n=6500, challenge=False):
    roles = rng.choice(['user', 'developer', 'server'], n, p=[.58, .24, .18])
    # Multiple events per host; ID groups are disjoint across train/validation/test.
    hosts_by_role = {r: [f'{r.upper()}-{i:03d}' for i in range(1, 141)] for r in ['user', 'developer', 'server']}
    hosts = [rng.choice(hosts_by_role[r]) for r in roles]
    p_mal = np.where(roles == 'server', .105, np.where(roles == 'developer', .115, .13))
    y = (rng.random(n) < p_mal).astype(int)
    proc_normal = {'user': ['chrome.exe', 'outlook.exe', 'teams.exe', 'powershell.exe', 'excel.exe'],
                   'developer': ['python.exe', 'git.exe', 'node.exe', 'powershell.exe', 'docker.exe'],
                   'server': ['svchost.exe', 'w3wp.exe', 'sqlservr.exe', 'powershell.exe', 'backup.exe']}
    proc = np.array([rng.choice(proc_normal[r]) for r in roles], dtype=object)
    proc[y == 1] = rng.choice(['powershell.exe', 'rundll32.exe', 'cmd.exe', 'mshta.exe', 'python.exe'], size=np.sum(y == 1))
    # Common administrative LOLBins are ALSO observed in benign activity; process name is not a label oracle.
    benign_lolbin = (y == 0) & (rng.random(n) < np.where((roles == 'developer') | (roles == 'server'), .16, .09))
    proc[benign_lolbin] = rng.choice(['rundll32.exe', 'cmd.exe', 'mshta.exe', 'powershell.exe'], size=np.sum(benign_lolbin))
    parent_choices = ['explorer.exe', 'services.exe', 'cmd.exe', 'winword.exe', 'w3wp.exe', 'devenv.exe']
    parents = rng.choice(parent_choices, n)
    dev = roles == 'developer'
    server = roles == 'server'
    hard_legit = (y == 0) & (dev | server) & (rng.random(n) < .34)
    stealth = (y == 1) & (rng.random(n) < .30)
    cmd_len = np.clip(rng.gamma(2, 36, n) + y * 125 + hard_legit * 130 - stealth * 112, 5, None).round().astype(int)
    prevalence = np.clip(rng.beta(4, 2, n) - y*.38 - hard_legit*.36 + stealth*.25, .002, .999).round(3)
    connections = rng.poisson(1.2 + dev*3 + server*5 + y*4 + hard_legit*4 - stealth*2)
    unique = np.maximum(0, rng.binomial(connections, np.clip(.24 + y*.22 + dev*.08, 0, .90)))
    outbound = np.round(rng.lognormal(10.7 + dev*.8 + server*.9 + y*.8 + hard_legit*.7, .9)).astype(int)
    executions = rng.poisson(8 + server*23 + dev*16 - y*5 + hard_legit*9) + 1
    unsigned = (rng.random(n) < np.clip(.055 + y*.60 + hard_legit*.44 - stealth*.39, 0, .98)).astype(int)
    unusual = (rng.random(n) < np.clip(.045 + y*.55 + hard_legit*.36 - stealth*.32, 0, .97)).astype(int)
    if challenge:
        # Distribution shift: new legitimate developer activity resembles malicious telemetry.
        shifted = (y == 0) & dev & (rng.random(n) < .70)
        connections[shifted] += rng.poisson(11, np.sum(shifted))
        cmd_len[shifted] += rng.integers(70, 180, np.sum(shifted))
        unsigned[shifted] = (rng.random(np.sum(shifted)) < .68).astype(int)
    frame = pd.DataFrame({
        'event_id': [f'EDR-{i:06d}' for i in range(1,n+1)], 'hostname': hosts, 'endpoint_type': roles,
        'process_name': proc, 'parent_process': parents, 'command_line_length': cmd_len,
        'process_prevalence': prevalence, 'network_connection_count': connections,
        'unique_destination_count': unique, 'outbound_bytes': outbound,
        'process_execution_frequency': executions, 'unsigned_binary': unsigned,
        'unusual_parent_child': unusual, 'random_trace_tag': rng.integers(10000, 99999, n),
        'malicious_label': y})
    return missing(rng, frame, ['process_prevalence', 'parent_process', 'outbound_bytes'], .045)


def triage(rng, n=3500, challenge=False):
    types = rng.choice(['phishing', 'malware', 'identity', 'edr', 'network', 'data_loss'], n)
    severity = rng.choice(['low', 'medium', 'high', 'critical'], n, p=[.28,.39,.25,.08])
    weight = pd.Series(types).map({'phishing': 5,'malware':14,'identity':11,'edr':13,'network':9,'data_loss':22}).to_numpy()
    sev = pd.Series(severity).map({'low':0,'medium':8,'high':18,'critical':32}).to_numpy()
    telemetry = rng.poisson(15, n) + 1
    hosts = rng.poisson(2.1, n) + 1
    historical = rng.poisson(12, n)
    workload = rng.integers(0, 14, n)
    complexity = np.clip(rng.normal(4.2, 2, n), .2, 10)
    # Nonlinear workload/complexity interaction challenges plain linear regression.
    minutes = (8 + weight + sev + .68*telemetry + 2.1*hosts - .42*historical
               + 2*workload + 1.45*complexity**2 + .85*workload*complexity
               + rng.normal(0, 16 if not challenge else 23, n))
    minutes = np.clip(minutes, 5, 360).round(1)
    if challenge:
        minutes += np.where(types == 'data_loss', 28, 0)
    frame = pd.DataFrame({
        'alert_id': [f'ALERT-{i:06d}' for i in range(1,n+1)], 'alert_type': types, 'alert_severity': severity,
        'telemetry_count': telemetry, 'related_host_count': hosts, 'historical_alert_frequency': historical,
        'analyst_workload': workload, 'investigation_complexity': complexity.round(2),
        'collection_day': rng.integers(1, 31, n), 'estimated_triage_minutes': minutes})
    return missing(rng, frame, ['investigation_complexity', 'historical_alert_frequency'], .035)


def network(rng, n=5000, challenge=False):
    roles = rng.choice(['user','developer','server'], n, p=[.55,.27,.18])
    sources = {r: [f'{r.upper()}-NET-{i:03d}' for i in range(1, 141)] for r in ['user', 'developer', 'server']}
    hosts = [rng.choice(sources[r]) for r in roles]
    dev = roles == 'developer'
    server = roles == 'server'
    anomaly = (rng.random(n) < .068).astype(int)
    unusual_legit = (anomaly == 0) & dev & (rng.random(n) < (.39 if challenge else .27))
    stealth = (anomaly == 1) & (rng.random(n) < .28)
    freq = rng.poisson(9 + dev*17 + server*25 + anomaly*28 + unusual_legit*26 - stealth*17) + 1
    dest = rng.poisson(2 + dev*4 + server*3 + anomaly*13 + unusual_legit*9 - stealth*7) + 1
    bsend = np.round(rng.lognormal(10.8 + dev*1 + server*.9 + anomaly*1.6 + unusual_legit*1.3 - stealth*.7, .9)).astype(int)
    brecv = np.round(rng.lognormal(11.8 + dev*.8 + server*1.2 + anomaly*.45 + unusual_legit*.7, .95)).astype(int)
    ports = rng.choice([80,443,53,22,3389,8080,8443], n, p=[.13,.54,.14,.07,.03,.07,.02])
    ports[anomaly.astype(bool) & ~stealth] = rng.choice([443, 8080, 8443, 3389], np.sum(anomaly.astype(bool) & ~stealth))
    duration = rng.gamma(2.1, 20, n) + anomaly*60 + unusual_legit*35 - stealth*25
    night = (rng.random(n) < np.clip(.08 + dev*.13 + server*.20 + anomaly*.43 + unusual_legit*.22 - stealth*.24, 0, .95)).astype(int)
    fails = rng.poisson(.5 + server*.65 + dev*.4 + anomaly*7.5 + unusual_legit*4.3 - stealth*4.5)
    if challenge:
        bsend[unusual_legit] = (bsend[unusual_legit] * 2).astype(int)
    frame = pd.DataFrame({
        'session_id': [f'NET-{i:06d}' for i in range(1,n+1)], 'source_host': hosts,
        'host_role': roles, 'connection_frequency': freq, 'destination_count': dest,
        'bytes_sent': bsend, 'bytes_received': brecv, 'destination_port': ports,
        'connection_duration': np.maximum(duration, 1).round(2), 'unusual_hour_activity': night,
        'failed_connection_count': np.maximum(fails, 0), 'random_trace_tag': rng.integers(10000,99999,n),
        'known_anomaly_label': anomaly})
    return missing(rng, frame, ['bytes_received', 'connection_duration'], .035)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=OUT)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        'phishing_emails.csv': phishing(np.random.default_rng(SEED+1)),
        'edr_events.csv': edr(np.random.default_rng(SEED+2)),
        'alert_triage.csv': triage(np.random.default_rng(SEED+3)),
        'network_activity.csv': network(np.random.default_rng(SEED+4)),
        'edr_events_challenge.csv': edr(np.random.default_rng(SEED+5), challenge=True),
        'alert_triage_challenge.csv': triage(np.random.default_rng(SEED+6), challenge=True),
        'network_activity_challenge.csv': network(np.random.default_rng(SEED+7), challenge=True),
    }
    leakage = data['phishing_emails.csv'].copy()
    flip = np.random.default_rng(SEED+18).random(len(leakage)) < .035
    leakage['post_investigation_decision'] = np.where(flip, 1-leakage['phishing_label'], leakage['phishing_label'])
    data['phishing_emails_leakage_challenge.csv'] = leakage
    for filename, frame in data.items():
        frame.to_csv(args.output_dir / filename, index=False)
        print(f'{filename:38} {frame.shape[0]:5} rows  {frame.shape[1]:2} columns')
    print(f'Seed base: {SEED}. All hostnames/domains are synthetic and non-routable.')


if __name__ == '__main__':
    main()
