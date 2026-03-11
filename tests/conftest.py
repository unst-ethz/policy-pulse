import pytest
import socket

def is_internet_available(host="8.8.8.8", port=53, timeout=3):
    """
    Check if there is an internet connection by trying to connect to Google's public DNS.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_internet: mark test as requiring internet access"
    )

def pytest_collection_modifyitems(config, items):
    if is_internet_available():
        return
    
    skip_internet = pytest.mark.skip(reason="No internet connection available")
    for item in items:
        if "needs_internet" in item.keywords:
            item.add_marker(skip_internet)
