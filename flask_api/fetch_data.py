import requests

url = "http://localhost/incident_system/api/incidents.php"

data = requests.get(url).json()
print(data)