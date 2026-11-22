"""Generate the Chinese results report from measured data; keep published numbers scoped to their tasks."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports'

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def pct(row,key):
    return f"{float(row[key])*100:.2f}%"

def md_table(headers,rows):
    return '\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in [headers,['---']*len(headers),*rows])

def main():
    primary=read_csv(ROOT/'results/replay-primary/summary.csv'); all_rows=read_csv(ROOT/'results/summary.csv'); rn50=read_csv(REPORT/'paper-comparison/eurosat_top1_comparison.csv')
    method_names={'visual':'Visual-Proto','zs':'ZS-CLIP','ensemble':'Prompt-Ensemble','fixed':'Fixed-Fusion','tip':'Tip-Adapter','gda-loo':'GDA-LOO','timo-loo':'TIMO-LOO','timo-proto':'TIMO-Proto','prolip':'ProLIP','lira':'LIRA-Proto'}
    scores={(r['requested_method'],r['condition']):r for r in primary}
    rows=[]
    for short,long in method_names.items():
        normal,wrong=(scores[short,c] for c in ('normal','wrong30'))
        rows.append([long,pct(normal,'accuracy'),pct(wrong,'accuracy'),pct(wrong,'macro_f1'),pct(wrong,'harm'),pct(wrong,'rescue')])
    paired=[r for r in read_json(ROOT/'results/replay-primary/paired.json') if r['metric']=='accuracy' and r['reference']=='timo-loo']
    curves=read_csv(REPORT/'candidate-outcomes.csv'); errors=[r for r in curves if r['condition']=='wrong30' and float(r['lambda_value'])==.5]
    avg=lambda k:mean(float(r[k]) for r in errors)
    main_rows={(r['dataset'],r['K']):r for r in all_rows if r['requested_method']=='lira' and r['condition']=='normal' and r['protocol']=='A' and r['variant']=='main'}
    calibrations={r['protocol']:r for r in all_rows if r['dataset']=='resisc45' and r['requested_method']=='lira' and r['K']=='4' and r['condition']=='normal' and r['variant']=='main'}
    controls=[r for r in all_rows if r['dataset']=='resisc45' and r['K']=='4' and r['protocol']=='A' and r['variant']=='ablation' and r['condition']=='wrong30']
    enc=read_json(REPORT/'encoder-check.json')['datasets']; archive=read_json(REPORT/'archive-metrics-audit.json'); formula=read_json(REPORT/'candidate-audit.json')
    native=read_csv(REPORT/'paper-comparison/native_replay_top1.csv'); audit=read_json(REPORT/'paper-comparison/verification.json')
    verified=read_json(REPORT/'fresh-metrics-audit.json'); acceptance=read_json(REPORT/'acceptance.json')
    assert acceptance['main_fresh']==verified['distinct_cells']==verified['expected_cells']==3600
    assert archive['passed'] and verified['passed'] and audit['resisc45_metrics_recomputed']==140
    cpu=read_csv(REPORT/'efficiency-same-cpu.csv')
    repeats=read_json(REPORT/'repeatability.json')
    diagnosis=read_json(ROOT/'results/diagnostics-check/run-status.json')
    assert diagnosis['failed_cells']==0
    sections=[
        '# LIRA-Proto 实验结果与文献对比',
        '2026 年 11 月 20 日。主要实验按研究计划预先指定的 NWPU-RESISC45、4-shot、严格支持集协议 A 执行。每次适配使用 180 张有标注支持图像，评测覆盖同一批 12,598 张查询图像。',
        '**主要结论**　正常文本和 30% 错误扩展文本下，LIRA-Proto 的 Top‑1 均为 69.51%，跨十个支持种子的样本标准差均为 1.48 个百分点。两种条件合计 20 次拟合都选了视觉回退。模型保留了视觉基线的全部正确分类，同时也保留了它的错误。严格预算下，TIMO-LOO 在这两种文本条件下的准确率分别为 77.78% 和 77.64%。',
        '## 1　同一标签预算下的十种方法',
        md_table(['方法','正常 Top‑1','错误 30% Top‑1','错误 30% Macro-F1','错误 30% Harm','错误 30% Rescue'],rows),
        '逐图的 Harm 与 Rescue 均以同一查询池为分母。TIMO-LOO 与固定融合产生了部分新增错误，同时修正了更多视觉分类错误。LIRA 与同头比较方法 TIMO-Proto 在此场景的配对 Top‑1 差为 0.00 个百分点，95% 配对区间为 [0.00, 0.00]。',
        md_table(['LIRA 减 TIMO-LOO','平均 Top‑1 差（百分点）','支持采样配对 95% 区间（百分点）'],[[{'normal':'正常','wrong30':'30% 错误'}[r['condition']],f"{100*r['difference']:.2f}",f"[{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]"] for r in paired]),
        '## 2　回退对分类有什么影响',
        f"把同一个有界模型固定在 λ=0.5，30% 错误描述下的查询 Top‑1 为 {100*avg('accuracy'):.2f}%，Harm 为 {100*avg('harm'):.2f}%，Rescue 为 {100*avg('rescue'):.2f}%。这个候选的支持集完整文本 Brier 相对 λ=0 平均增加 {avg('full_brier_delta'):.5f}，但查询准确率平均高 {100*avg('accuracy_minus_visual'):.2f} 个百分点。该差异刻画了保守准入的代价。种子 101 中，回退保住了一个本会由棒球场改成网球场的判断，也错过了一个可由跑道纠正为飞机的判断。",
        md_table(['核心消融，30% 错误文本','Top‑1','Harm','Rescue'],[[r['requested_method'],pct(r,'accuracy'),pct(r,'harm'),pct(r,'rescue')] for r in controls]),
        '对研究假设而言，H1 在与固定融合比较时观察到误伤减少，与同头 TIMO-Proto 比较时两者均为零误伤。H2 的几何下界及稳定性判据通过逐折和逐图检查，主场景中选出的零位移使其独立准确率贡献无法从这一对照估计。H3 的严格预算路径已跑通，主场景上的相对准确率增益为零。H4 的本机可运行性由三套真实图像、实际运行时间和峰值内存记录支持。',
        '## 3　三套数据、五档支持预算与概率校准',
        md_table(['LIRA，正常文本','1-shot','2-shot','4-shot','8-shot','16-shot'],[[label,*[pct(main_rows[d,str(k)],'accuracy') for k in (1,2,4,8,16)]] for d,label in [('eurosat','EuroSAT RGB'),('ucmerced','UC Merced'),('resisc45','NWPU-RESISC45')]]),
        '1-shot 按计划直接选择视觉原型。三个数据集在 2-shot 正常描述下的十个支持种子都选择非零 λ。准确率随预算和数据集而变化，λ=0 是一次支持采样内部的风险选择结果。',
        md_table(['RESISC45 4-shot 正常','总标签','Top‑1','Brier','NLL','ECE'],[[p,str(45*(4 if p=='A' else 24)),pct(calibrations[p],'accuracy'),*[f"{float(calibrations[p][('' if p=='A' else 'cal_')+k]):.5f}" for k in ('brier','nll','ece')]] for p in ('A','B','B-temp')]),
        '后两行使用额外标签拟合温度。B 中每类 10 个模型选择标签、10 个温度标签，B-temp 使用每类 20 个温度标签。',
        '## 4　与三篇参考文献的比较',
        '**附件 2，TIMO。** 论文表 1 的 4-shot TIMO 为 71.20%，是 11 个数据集、每个数据集 3 个种子的平均。为进行 EuroSAT 对照，本工程使用相同原始 CoOp 查询清单（8,100 张）、RN50、五档 K 和种子 1、2、3。作者列取自固定上游归档 `outputs/TIMO.txt` 和 `outputs/TIMO_S.txt`，本地列由保存的逐图 logits 重算。',
        md_table(['K','作者 TIMO','作者 TIMO-S','本地 Visual','本地 TIMO-Proto','本地 TIMO-LOO','本地 LIRA'],[[r['K'],*[f"{float(r[k]):.2f}%" for k in ('timo_top1_pct','timo_s_top1_pct','visual_top1_pct','timo_proto_top1_pct','timo_loo_top1_pct','lira_top1_pct')]] for r in rn50]),
        '原始 TIMO 使用 CuPL 提示、图像增强和有标签验证集。4-shot 时，LIRA 对严格预算 TIMO-LOO 的差为 −6.67 个百分点，对作者 TIMO 的差为 −11.86 个百分点。两个差值从未舍入的三种子均值计算。原生配置单种子实跑如下。',
        md_table(['RN50 本机作者配置，K=4、种子 1','正确数 / 查询数','Top‑1','额外验证标签'],[[r['method'].upper() if r['method']=='timo' else 'ProLIP',f"{r['correct']} / {r['query_n']}",f"{float(r['top1_pct']):.2f}%",r['validation_labels']] for r in native]),
        '**附件 1，MEET / CAT。** 论文 PDF 第 10 页正文与第 11 页表 II 在 MEET 细粒度场景分类上报告 CAT（Swin-Huge）的 OA 为 97.74%、BA 为 89.37%。其中 OA 是全部场景的正确率，BA 是每类召回率的平均。该实验使用 MEET 类别和监督设置。表 VI 中的 NWPU 数值对应城市功能区映射，45 类 RESISC45 少样本分类使用另一套真值，因此该表无法估计两种方法的分类性能差。',
        '**附件 3，遥感视觉语言综述。** 表 II 报告 RemoteCLIP（ViT-L/14）零样本 Top‑1，EuroSAT 为 59.94%，RESISC45 为 79.84%。本项目冻结 ViT-B/32 的零样本 ZS-CLIP 在自己的查询池上分别为 41.35% 和 57.82%。这组数字说明模型、预训练来源和划分的差异会影响比较。严格的 LIRA 校正效果由第 1 节同骨干、同图像、同标签预算的对照估计。',
        '## 5　运行规模与核查记录',
        f"主矩阵在新环境重新拟合 {verified['distinct_cells']:,} 个单元，覆盖 3 套数据、5 个 K、10 组支持抽样和两种主文本。ProLIP 在每个条件有三个优化种子。既有正式档案另有 {archive['files']:,} 个逐单元文件完成哈希与指标复核，去重后为 {archive['distinct_cells']:,} 个实验组合，包含相同的主矩阵以及三套数据的 4-shot 额外文本、校准、消融和预定敏感性。十个方法的历史诊断由原始预测和查询真值重新计算，归档与本次重跑的相同实验键只计一次。",
        f"同配置逐项配对复算后，{repeats['same_top1_harm_rescue']:,} / {repeats['paired_fits']:,} 组的 Top‑1、Harm 和 Rescue 同时相同。三套数据的支持种子 101 另重新拟合 {diagnosis['completed_fits']} 个诊断单元，覆盖 4-shot 文本类型、核心消融、独立校准和指定主场景的参数敏感性。",
        f"三套数据合计 60,600 张原图。主协议重新编码候选池与查询池合计 {sum(v['images'] for v in enc.values()):,} 张，并重新编码 36 组文本条件。398 个 CLIP 模型张量与固定原始检查点逐项对照。主场景独立显式删一完成 {formula['folds']:,} 个图像与文本折、{formula['checked_risk_entries']} 个风险值，最大绝对差 {formula['max_abs_risk_error']:.2e}。附加的 EuroSAT RN50 实验独立核验 60 组本地结果和 30 行作者记录。",
        md_table(['重新编码','图像数','批大小','图像阶段秒','图像投影后缓存最大绝对差','峰值 GPU 张量分配 MiB'],[[d,v['images'],v['batch'],f"{v['image_seconds']:.2f}",f"{v['max_abs_z']:.2e}",f"{v['gpu_peak_allocated']/2**20:.1f}"] for d,v in enc.items()]),
        f"首个数据集 UC Merced 从启动、载入编码器、检查权重、读取缓存、从原图编码、对照文本直至完成一组 4-shot 评估约 {enc['ucmerced']['elapsed_from_process_start_seconds']:.1f} 秒。表中图像阶段含缓存读取、哈希和差异检查。单进程峰值工作集最高 {max(r['process_peak_ws_bytes'] for r in enc.values())/2**30:.2f} GiB。额外同设备对照将十种方法全部放在 CPU，固定同一次支持抽样、2 个线程，各预热一遍后计时三遍；LIRA 的平均拟合时间为 {mean(float(r['fit_seconds']) for r in cpu if r['method']=='lira'):.3f} 秒。具体耗时取决于设备和同时运行的任务。",
        '本次技术范围是计划的 CLIP ViT-B/32 主干、RN50 作者配置、五档少样本主矩阵和 4-shot 扩展诊断。RemoteCLIP 的共同协议换骨干实验是计划列出的可选扩展。公开数据未提供可核验的原始地理分组，因此区间和结论对应这些封存查询池及十组支持采样。',
        '[逐图审计](reports/fresh-metrics-audit.json) · [历史诊断审计](reports/archive-metrics-audit.json) · [内部风险与外部效果](reports/candidate-outcomes.csv) · [同设备效率](reports/efficiency-same-cpu.csv) · [论文对照核验](reports/paper-comparison/verification.json) · [文件与方法说明](项目文件与复现说明.md)'
    ]
    (ROOT/'实验结果与文献对比.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')

if __name__=='__main__':main()
