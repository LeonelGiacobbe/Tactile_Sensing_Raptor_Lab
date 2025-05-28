from pymodbus.client import ModbusTcpClient

# Connect to the gripper (replace IP with your actual device IP)
client = ModbusTcpClient("192.168.1.10", port=502)
client.connect()

# Use the correct syntax: pass `slave` instead of `unit`
result = client.read_holding_registers(address=0, count=1, slave=1)

if result.isError():
    print("Error reading status register")
else:
    while True:
        gOBJ = result.registers[0]
        if gOBJ == 0x00:
            print("No object detected")
        elif gOBJ == 0x01:
            print("Object detected during opening")
            break
        elif gOBJ == 0x02:
            print("Object detected during closing")
            break

client.close()
