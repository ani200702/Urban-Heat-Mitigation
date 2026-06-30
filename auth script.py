import ee

try:
    # This will open a browser window for you to log in if needed.
    ee.Authenticate()
except Exception as exc:
    print(f"Authentication prompt finished with: {exc}")

try:
    # This will initialize Earth Engine for your specific project.
    ee.Initialize(project='isro-urban-heat-mitigation')
    print("Authentication successful!")
except Exception as exc:
    print("Earth Engine initialization failed.")
    print("Enable the Earth Engine API for your Google Cloud project and wait a few minutes before retrying.")
    print("Open: https://console.developers.google.com/apis/api/earthengine.googleapis.com/overview?project=isro-urban-heat-mitigation")
    print(f"Details: {exc}")