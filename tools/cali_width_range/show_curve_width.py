import json
import matplotlib.pyplot as plt

example_gripper_widths = [0, 200, 400, 600, 800, 1000]
example_aruco_widths = [0.05342744446896867, 0.06899043027148796, 0.08455341607400725, 0.09946971260254554, 0.1145028226793132, 0.12981197752954984]  

with open('/home/drj/codehub/ViTaMIn-B/assets/cali_width_result/width_calibration.json', 'r') as f:
    json_data = json.load(f)

slope = json_data['slope']
intercept = json_data['intercept']

point_x_y = list(zip(example_gripper_widths, example_aruco_widths))

plt.figure(figsize=(10, 6))

plt.scatter(example_gripper_widths, example_aruco_widths, color='red', s=50, label='Data Points', zorder=3)

x_line = range(0, 1100, 10)
y_line = [slope * x + intercept for x in x_line]
plt.plot(x_line, y_line, color='blue', linewidth=2, label=f'Fitted Line: y = {slope:.2e}x + {intercept:.4f}')

plt.title('Gripper Width vs ArUco Width Calibration Relationship', fontsize=14)
plt.xlabel('Gripper Width', fontsize=12)
plt.ylabel('ArUco Width', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

for i, (x, y) in enumerate(point_x_y):
    plt.annotate(f'({x}, {y:.4f})', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("Coordinates of six points:")
for i, (x, y) in enumerate(point_x_y):
    print(f"Point {i+1}: ({x}, {y:.6f})")

print(f"\nFitted line equation: y = {slope:.6e}x + {intercept:.6f}")
print(f"Slope: {slope:.6e}")
print(f"Intercept: {intercept:.6f}")
