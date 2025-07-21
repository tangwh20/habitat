from scipy.spatial.transform import Rotation as R
import numpy as np


"""
action: FORWARD
Destination, distance: 5.038557, theta(radians): -0.01
position:  [0.7669383  0.08441114 9.30006   ]
rotation:  quaternion(0.971329987049103, 0, -0.237729996442795, 0
action: LEFT
Destination, distance: 5.038557, theta(radians): -0.19
position:  [0.7669383  0.08441114 9.30006   ]
rotation:  quaternion(0.988354563713074, 0, -0.152168571949005, 0)
action: FORWARD
Destination, distance: 4.793059, theta(radians): -0.19
position:  [0.84213656 0.08441114 9.061638  ]
rotation:  quaternion(0.988354563713074, 0, -0.152168571949005, 0)
action: LEFT
Destination, distance: 4.793059, theta(radians): -0.37
position:  [0.84213656 0.08441114 9.061638  ]
rotation:  quaternion(0.997855961322784, 0, -0.0654487609863281, 0)
action: FORWARD
Destination, distance: 4.560824, theta(radians): -0.39
position:  [0.8747908  0.08441114 8.81378   ]
rotation:  quaternion(0.997855961322784, 0, -0.0654487609863281, 0)
"""

# r = R.from_quat([
#     [0.971329987049103, 0, -0.237729996442795, 0],
#     [0.988354563713074, 0, -0.152168571949005, 0],
#     [0.997855961322784, 0, -0.0654487609863281, 0]
# ])
r = R.from_euler('yxz', [[ 152.49476177,    0.,          180.        ],
 [ 162.49476126,    0.,         -180.        ],
 [ 172.49475998,    0.,         -180.        ]], degrees=True)

r1 = R.from_quat([0.971329987049103, 0, -0.237729996442795, 0])
# r2 = R.from_quat([0.971329987049103, 0, -0.237729996442795, 0])
# r3 = R.from_quat([0.946915447711945, 0, -0.32148277759552, 0])

print(r.as_euler('yxz', degrees=True))
breakpoint()
print(r1.as_euler('xyz', degrees=True))
# print(r2.as_euler('xyz', degrees=True))
# print(r3.as_euler('xyz', degrees=True))

r1.as_quat
print(r1.as_matrix())

# pos = np.array([0.26693833, 0.08441114, 9.30006   ])
pos = np.array([0.7669383, 0.08441114, 9.30006   ])
dist = np.array([0.25*np.cos(np.radians(10)), 0.25*np.sin(np.radians(10))])

yaw = r1.as_euler('yxz', degrees=True)[0]
pos[0] += (dist[0] * np.sin(np.radians(yaw)) + dist[1] * np.cos(np.radians(yaw)))
pos[2] += (dist[0] * np.cos(np.radians(yaw)) - dist[1] * np.sin(np.radians(yaw)))
print(pos)

breakpoint()
print(r1.apply(dist) + pos)