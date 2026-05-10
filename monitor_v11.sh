#!/bin/bash
echo "========================================="
echo "V11训练进度监控"
echo "========================================="
echo ""

# 检查进程
if pgrep -f "training.py -v v11" > /dev/null; then
    echo "✅ 训练进程运行中"
    echo ""
    
    # 显示最新日志
    echo "最新训练进度:"
    echo "-----------------------------------------"
    tail -20 /home/airst/Workspace/vnpy/log/run_v11.log | grep -E "\[|===|训练|Factor|MLP|回测"
    echo ""
    
    # 检查是否已有回测结果
    if ls /home/airst/Workspace/vnpy/core/alpha_db/backtest/*v11*.json 1> /dev/null 2>&1; then
        echo "✅ 已生成回测结果:"
        ls -lht /home/airst/Workspace/vnpy/core/alpha_db/backtest/*v11*.json | head -3
    else
        echo "⏳ 等待回测结果生成..."
    fi
else
    echo "❌ 训练进程未运行"
    echo ""
    echo "检查日志文件:"
    ls -lht /home/airst/Workspace/vnpy/log/run_v11.log 2>/dev/null
fi
