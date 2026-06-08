import requests

url = "https://incidentmap.kesug.com/get_incidents.php"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

response = requests.get(url, headers=headers, timeout=20)

print("STATUS CODE:", response.status_code)
print("RESPONSE SAMPLE:")
print(response.text[:500])