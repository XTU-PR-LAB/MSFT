import json
import numpy as np

# 从 JSON 文件中读取结果
with open("results.json", "r") as f:
    results = json.load(f)

# 计算所有场景的平均值的平均值
overall_average_psnr = np.mean(results["psnr"])
overall_average_lpips = np.mean(results["lpips"])
overall_average_ssim = np.mean(results["ssim"])
psnr_value = 10 ** (-overall_average_psnr/ 10)
    
# 计算sqrt(1 - SSIM)
ssim_value = np.sqrt(1 - overall_average_ssim)

# 计算几何平均值
avg_mean = (psnr_value * ssim_value * overall_average_lpips) ** (1/3)

print("Overall Average PSNR: ", overall_average_psnr)
print("Overall Average LPIPS: ", overall_average_lpips)
print("Overall Average SSIM: ", overall_average_ssim)
print("Overall Average : ", avg_mean)
