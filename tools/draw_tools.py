"""
本模块是绘图函数的汇总
"""
import matplotlib.pyplot as plt
import numpy as np
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def draw_hist(data:list,image_path:str,normed=False) -> None:
    """
    绘制data元素分布的直方图，图的横轴表示一个范围区间，区间的长度会根据data的最大值进行调整一般会控制区间个数在30个以内，纵轴是data落于横轴区间的元素个数。
    绘制的图形写入image_path
    """
    import math
    m = max(data)
    range_max = 0
    bin_num = 10
    # 根据data的最大值调整区间范围和长度
    if m <= 1:
        data_range = (0,1)
        range_max = 1
    else:
        m = round(m)
        n = len(str(m)) - 2
        base = math.pow(10,n)
        if m/base > 30:
            base = base * 10
        bin_num = math.ceil(m/base)
        range_max = bin_num*base
        print(f"data max {m},base {base}, range max : {range_max}")
        data_range = (0,range_max)
    # 开始绘图
    figure = plt.figure(figsize=(9,16),dpi=400)
    plt.xlabel("value")
    plt.ylabel("number")
    colors = plt.get_cmap("Reds")
    n, bins, patches = plt.hist(x=data,bins=bin_num,range=data_range,align="mid",color="Red",density=normed) # hist里所有的bin默认同颜色
    print(f"bins:\n{bins}")
    # 为bin设置不同的颜色
    for c,p in zip(bins,patches):
        p.set_facecolor(colors(c/range_max))
    for i in range(len(n)):
        plt.text(bins[i]+range_max/(bin_num*2), n[i]*1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    xt = [(range_max/bin_num)*i for i in range(bin_num+1)]
    plt.xticks(xt)
    figure.savefig(image_path)
    return

def draw_corr_hist(data:list,image_path:str,normed=False) -> None:
    """
    绘制data元素分布的直方图，图的横轴表示一个范围区间，区间的长度会根据data的最大值进行调整一般会控制区间个数在30个以内，纵轴是data落于横轴区间的元素个数。
    绘制的图形写入image_path
    """
    import math
    m = max(data)
    range_max = 2
    bin_num = 10
    # 根据data的最大值调整区间范围和长度
    data_range = (0,2)
    # 开始绘图
    figure = plt.figure(figsize=(9,16),dpi=400)
    plt.xlabel("value")
    plt.ylabel("number")
    colors = plt.get_cmap("Reds")
    n, bins, patches = plt.hist(x=data,bins=bin_num,range=data_range,align="mid",color="Red",density=normed) # hist里所有的bin默认同颜色
    print(f"bins:\n{bins}")
    # 为bin设置不同的颜色
    for c,p in zip(bins,patches):
        p.set_facecolor(colors(c/range_max))
    for i in range(len(n)):
        plt.text(bins[i]+range_max/(bin_num*2), n[i]*1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    xt = [(range_max/bin_num)*i for i in range(bin_num+1)]
    plt.xticks(xt)
    figure.savefig(image_path)
    return


def draw_plots(data,image_path):
    """
    绘制折线图
    data是一个字典内容是{"label:value"} value is a dict: {cir:[task_id1,task_id2...]}
    """
    # data format: {"label:value"} value is a dict: {cir:[task_id1,task_id2...]}
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(len(v))
        xs.append(x)
        y = v
        ys.append(y)
    fig = plt.figure(figsize=(9,6),dpi=400)
    plt.xlabel("Cirs")
    plt.ylabel("number of task")
    plt.title(image_path.split(".")[0])
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        plt.plot(x,y)
        for xz,yz in zip(x,y):
            plt.text(xz,yz,yz)
    plt.legend(labels,loc="upper left")
    fig.savefig(image_path)
    return

def draw_plots_percent(data,color,image_path):
    """
    绘制折线图
    data是一个字典内容是{"label:value"} value is a dict: {cir:[task_id1,task_id2...]}
    """
    # data format: {"label:value"} value is a list of all values
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(len(v))
        xs.append(x)
        y = v
        ys.append(y)
    fig = plt.figure(figsize=(18,12),dpi=400)
    plt.xlabel("Cirs",fontsize='large')
    plt.ylabel("percent:%",fontsize='large')
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    
    plots = []
    for i in range(len(data.keys())):
        x = xs[i]
        y = ys[i]
        p, = plt.plot(x,y,marker='o',color=color[labels[i]],linewidth=2)
        plots.append(p)
        for xz,yz in zip(x,y):
            plt.text(xz,yz+0.5,yz,fontsize='large')
        plt.text(10+0.1,y[10],labels[i],fontsize="x-large",color=color[labels[i]])
    plt.legend(handles=plots,labels=labels,loc="upper left")#填best自动找到最好的位置
    plt.xticks(range(14),[str(i) for i in range(14)])
    fig.savefig(image_path)
    return

def draw_bars(data,image_path):
    """
    绘制方形图
    data是一个字典内容是{"percent:numbers"}
    """
    xs = []
    ys = []
    for k,v in data.items():
        xs.append(k)
        ys.append(v)
    fig = plt.figure(figsize=(5,3),dpi=400)
    plt.xlabel("numbers")
    plt.ylabel("correct percent")
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    plt.bar(x=xs,height=ys,color="blue")
    # for xt,yt in zip(xs,ys):
    #     plt.text(xt,yt,yt,va="bottom",ha="center")
    fig.savefig(image_path)
    return

def draw_scatters(data,image_path):
    """
    绘制散点图
    data是一个字典内容是{"percent:numbers"}
    """
    # plt.style.use("fivethirtyeight")
    xs = []
    ys = []
    for k,v in data.items():
        xs.append(k)
        ys.append(v*100)
    fig = plt.figure(figsize=(6,4),dpi=400)
    plt.xlabel("通过测试用例的代码数量")
    plt.ylabel("测试用例正确率:%")
    plt.yticks([0.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0])
    # plt.xticks(range(0,300,50))
    title = image_path.split("/")[-1].split(".")[0]
    # plt.title(title)
    plt.grid(True,linestyle="--",alpha=0.5)
    plt.scatter(x=xs,y=ys,s=5,color="blue")
    # for xt,yt in zip(xs,ys):
    #     plt.text(xt,yt,yt,va="bottom",ha="center")
    fig.savefig(image_path)
    return

def draw_plots_mean_std(data,data2,color,image_path):
    # data format: {"label:value"} value is a list of all values
    xs = []
    ys = []
    labels = []
    for k,v in data.items():
        labels.append(k)
        x = range(len(v))
        xs.append(x)
        y = v
        ys.append(y)
    x2 = data2["x"]
    mean_y = data2["mean"]
    std_y = data2["std"]
    xs.append(x2)
    ys.append(mean_y)
    labels.append(data2["label"])
    fig = plt.figure(figsize=(18,12),dpi=400)
    plt.xlabel("Cirs",fontsize='large')
    plt.ylabel("percent:%",fontsize='large')
    title = image_path.split("/")[-1].split(".")[0]
    plt.title(title)
    
    plots = []
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        p, = plt.plot(x,y,marker='o',color=color[labels[i]],linewidth=2)
        plots.append(p)
        for xz,yz in zip(x,y):
            plt.text(xz,yz+0.5,yz,fontsize='large')
        plt.text(10+0.1,y[10],labels[i],fontsize="x-large",color=color[labels[i]])
    plt.legend(handles=plots,labels=labels,loc="upper left")#填best自动找到最好的位置
    plt.xticks(range(14),[str(i) for i in range(14)])
    plt.fill_between(x2,mean_y-std_y,mean_y+std_y,alpha=0.3,color="blue")
    fig.savefig(image_path)
    return
