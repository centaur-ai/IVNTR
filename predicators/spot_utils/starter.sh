echo "Checking WIFI Connection..."
echo ping 192.168.80.3
python3 -m bosdyn.client 192.168.80.3 id

export BOSDYN_CLIENT_USERNAME=admin
export BOSDYN_CLIENT_PASSWORD=8bjvkcjtghki
# export BOSDYN_CLIENT_PASSWORD=jtidtt7fbxh5

python3 hello_spot.py 192.168.80.3