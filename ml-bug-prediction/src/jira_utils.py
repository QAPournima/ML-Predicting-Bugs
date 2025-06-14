from jira import JIRA
import pandas as pd

def fetch_jira_bugs(jira_url, email, api_token, project_key):
    jira = JIRA(server=jira_url, basic_auth=(email, api_token))
    jql = f'project = {project_key} AND issuetype = Bug'
    issues = jira.search_issues(jql, maxResults=1000)
    data = []
    for issue in issues:
        data.append({
            'key': issue.key,
            'summary': issue.fields.summary,
            'status': issue.fields.status.name,
            'created': issue.fields.created,
            'resolved': getattr(issue.fields.resolutiondate, 'isoformat', lambda: None)(),
            'assignee': getattr(issue.fields.assignee, 'displayName', None),
            'reporter': getattr(issue.fields.reporter, 'displayName', None),
            'priority': getattr(issue.fields.priority, 'name', None),
            # Add more fields as needed
        })
    return pd.DataFrame(data)