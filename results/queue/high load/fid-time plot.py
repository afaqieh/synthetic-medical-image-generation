import matplotlib.pyplot as plt

# Diffusion steps (same for both resolutions)
steps = [10, 20, 30, 40, 50]

# 512x512 results
latency_512 = [0.537, 0.80655, 1.09376, 1.39018, 1.7004]
fid_512     = [161.3278, 159.8, 152.03, 159.7, 150.727]

# 256x256 results
latency_256 = [0.356779, 0.5979617, 0.84998, 1.1902, 1.44889]
fid_256     = [406.71, 282.52, 254.33, 234.16, 239.56]

plt.figure(figsize=(8, 8))

# -------- Top plot: Latency vs Diffusion Steps --------
ax1 = plt.subplot(2, 1, 1)
ax1.plot(steps, latency_512, marker='o', label='512×512')
ax1.plot(steps, latency_256, marker='o', label='256×256')

ax1.set_title('Latency vs Diffusion Steps')
ax1.set_xlabel('Diffusion steps')
ax1.set_ylabel('Latency (s)')
ax1.legend()
ax1.grid(True)

# -------- Bottom plot: FID vs Diffusion Steps --------
ax2 = plt.subplot(2, 1, 2)
ax2.plot(steps, fid_512, marker='o', label='512×512')
ax2.plot(steps, fid_256, marker='o', label='256×256')

ax2.set_title('FID vs Diffusion Steps')
ax2.set_xlabel('Diffusion steps')
ax2.set_ylabel('FID')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('figure.png')
plt.show()