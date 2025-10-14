#!/bin/bash
# Быстрая проверка всех исправлений перед обучением

set -e

echo "================================================================================"
echo "БЫСТРАЯ ПРОВЕРКА ПЕРЕД ОБУЧЕНИЕМ"
echo "================================================================================"

cd "$(dirname "$0")/.."

echo ""
echo "1. Проверка окружения..."
if ! command -v conda &> /dev/null; then
    echo "   ❌ Conda не найден!"
    exit 1
fi
echo "   ✅ Conda установлен"

if ! conda env list | grep -q "roma"; then
    echo "   ❌ Conda env 'roma' не найден!"
    exit 1
fi
echo "   ✅ Conda env 'roma' существует"

echo ""
echo "2. Проверка PyTorch и CUDA..."
conda run -n roma python -c "
import torch
print(f'   ✅ PyTorch {torch.__version__}')
if torch.cuda.is_available():
    print(f'   ✅ CUDA доступна ({torch.cuda.device_count()} GPU)')
else:
    print('   ⚠️  CUDA недоступна (будет использован CPU)')
"

echo ""
echo "3. Проверка данных..."
if [ ! -f "/home/sema/radar/datasets/umbra_tiles/pairs_train.json" ]; then
    echo "   ❌ Тренировочные данные не найдены!"
    exit 1
fi
echo "   ✅ pairs_train.json существует"

if [ ! -f "/home/sema/radar/datasets/umbra_tiles/pairs_val.json" ]; then
    echo "   ❌ Валидационные данные не найдены!"
    exit 1
fi
echo "   ✅ pairs_val.json существует"

echo ""
echo "4. Проверка файлов кода..."
files=(
    "romatch/datasets/umbra.py"
    "romatch/benchmarks/umbra_dense_benchmark.py"
    "experiments/train_roma_umbra.py"
    "experiments/benchmark_umbra.py"
)

for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "   ❌ Файл не найден: $file"
        exit 1
    fi
done
echo "   ✅ Все необходимые файлы на месте"

echo ""
echo "5. Проверка исправлений в коде..."
conda run -n roma python << 'EOF'
import sys
sys.path.insert(0, '/home/sema/radar/RoMa')

from romatch.datasets.umbra import UmbraScene
import json

# Загружаем валидационные данные
with open('/home/sema/radar/datasets/umbra_tiles/pairs_val.json') as f:
    scene_info = json.load(f)

# Проверка 1: Матрицы K и T
dataset = UmbraScene(scene_info, image_size=640, use_horizontal_flip_aug=False, shake_t=0)
sample = dataset[0]

K = sample['K1'].numpy()
T = sample['T_1to2'].numpy()

# Критичные проверки
assert K[0, 0] == 1.0, f"K[0,0]={K[0,0]} должно быть 1.0!"
assert abs(T[1, 3]) < 2, f"T[1,3]={T[1,3]} должно быть normalized!"
print("   ✅ Матрицы K и T корректны")

# Проверка 2: Аугментации
dataset_aug = UmbraScene(scene_info, image_size=640, use_horizontal_flip_aug=True, shake_t=32)
assert hasattr(dataset_aug, 'horizontal_flip'), "Нет метода horizontal_flip!"
assert dataset_aug.shake_t == 32, f"shake_t={dataset_aug.shake_t}, должно быть 32!"
print("   ✅ Аугментации добавлены")

# Проверка 3: Бенчмарк
from romatch.benchmarks.umbra_dense_benchmark import UmbraDenseBenchmark
benchmark = UmbraDenseBenchmark(scene_info, image_size=640, num_samples=5)
assert len(benchmark.dataset) > 0, "Датасет пустой!"
print("   ✅ Бенчмарк работает")

print("\n   🎉 ВСЕ ТЕХНИЧЕСКИЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
EOF

echo ""
echo "6. Проверка свободного места..."
free_space=$(df -h . | awk 'NR==2 {print $4}')
echo "   ℹ️  Свободно: $free_space"

echo ""
echo "================================================================================"
echo "✅ ГОТОВО К ОБУЧЕНИЮ!"
echo "================================================================================"
echo ""
echo "Запустите обучение:"
echo ""
echo "  torchrun --nproc_per_node=2 experiments/train_roma_umbra.py \\"
echo "      --train_data /home/sema/radar/datasets/umbra_tiles/pairs_train.json \\"
echo "      --val_data /home/sema/radar/datasets/umbra_tiles/pairs_val.json \\"
echo "      --checkpoint_dir workspace/checkpoints_umbra \\"
echo "      --gpu_batch_size 4"
echo ""

