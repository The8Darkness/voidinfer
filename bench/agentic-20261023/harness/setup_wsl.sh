set -e
python3 -m venv ~/agentbench/.venv 2>/dev/null || { sudo -n apt-get install -y python3-venv; python3 -m venv ~/agentbench/.venv; }
~/agentbench/.venv/bin/pip -q install pytest cmake
~/agentbench/.venv/bin/pytest --version
~/agentbench/.venv/bin/cmake --version | head -1
make --version | head -1
