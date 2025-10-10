import argparse
import json
import logging
import base64
import os
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- 新增 ---: 导入 OpenAI 库用于本地接口调用
try:
    from openai import OpenAI
except ImportError:
    # 如果用户不使用本地模式，这个库不是必需的
    OpenAI = None 

# 使用 tqdm 显示进度条
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print("tqdm is not installed, progress bar will not be shown. "
              "Install it with: pip install tqdm")
        return iterable

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Prompt 设计：合并领域、意图和槽位列表 ---

# 1. 领域和意图列表 (来自 slu_ic.py)
DOMAIN_INTENT_LIST = """
- 车载控制
    - 车机控制
    - 车身控制
    - 提供信息
- 地图
    - 导航
    - 提供地址
    - 查询路况
    - 查询定位
    - 查询路程
    - 查询前方路线
    - 导航路线规划
    - 设置常用地址
    - 导航到常用地址
    - 沿途搜索
    - 周边搜索
    - 增加途经点
    - 删除途经点
    - 地图操作
    - 上报事件
    - sys.确认
    - sys.取消
    - sys.用户选择
    - 限速查询
    - 设置目的地
    - 查询目的地
    - 修改途经点
    - 收藏
    - 取消收藏
- 音乐
    - 播放音乐
    - 播放控制
    - 查询音乐信息
    - 播放收藏
    - 播放列表
    - 播放历史
    - 新手引导
    - sys.用户选择
    - sys.确认
    - sys.取消
- 打电话
    - 拨打电话
    - 电话控制
    - 接听电话
    - 挂断电话
    - sys.确认
    - sys.取消
    - 查询信息
    - sys.电话选择
    - 拨打黄页号码
- 收音机
    - 播放电台
    - 播放控制
    - 播放收藏
    - 收音机控制
- 天气
    - 查询天气
    - 查询气象
    - 查询温度
    - 查询湿度
    - 查询风力
    - 查询风向
    - 查询空气质量
    - 查询紫外线
    - 查询日出日落
    - 查询活动
    - 查询装备
    - 穿衣推荐
    - 新手引导
    - 查询日期
    - 查询城市
    - 查询场景
    - 查询护肤品
    - 查询能见度
    - 查询指数
    - 查询降水量
    - 查询降雪量
    - sys.确认
    - sys.取消
    - sys.用户选择
- 影视
    - 播放影视
    - 播放控制
    - 播放收藏
    - 播放列表
    - 播放历史
    - sys.确认
    - sys.取消
    - sys.用户选择
    - 查询影视信息
- 播放控制
    - 播放控制
"""

# 2. 槽位列表 (来自 slu_sf.py)
SLOT_LIST = """
- 地图-__act__: 地图场景下的用户意图，例如"request"。
- 地图-__tgt__: 地图场景下用户意图的目标，例如前方路况、前方路线、剩余距离。
- 地图-poi修饰: 兴趣点（POI）的修饰词，用于更精确地描述位置，例如南山区、深圳南山、目的地。
- 地图-poi名称: 兴趣点（POI）的通用名称，例如学府路、当前点、目的地。
- 地图-poi目标: 兴趣点（POI）的具体目标，通常是专有名词，例如万达广场、北京、海底捞。
- 地图-poi类型: 兴趣点（POI）的类别，例如徽菜、杏仁瓦片、笋唤春生石榴包。
- 地图-sys.序列号: 系统定义的序列号或顺序，例如第一个途经点、第二个途经点。
- 地图-sys.指代: 系统定义的指代词，用于引用上文提到的内容，例如poi修饰。
- 地图-sys.页码: 系统定义的页码，例如上一页。
- 地图-事件: 导航过程中发生的交通事件，例如逃逸、逆向行驶、龟速。
- 地图-充电功率: 电动汽车充电桩的功率类型，例如快充、支持快充的、能快充的。
- 地图-充电品牌: 充电桩的品牌，例如特来电。
- 地图-地图尺寸: 地图的缩放级别，例如最小、调小。
- 地图-对象: 地图功能中所操作的对象，例如电子眼播报、详细信息、途经点。
- 地图-导航视角: 导航时的地图显示视角，例如北向上、北方向朝上、车头向上。
- 地图-导航道路位置: 导航时车辆所在的道路位置，例如主路、桥上、桥下。
- 地图-操作: 对地图进行的操作，例如导航、显示、查找。
- 地图-模式: 地图的显示模式，例如夜间地图、日间地图、黑夜模式。
- 地图-电站筛选条件: 筛选充电站的条件，例如空闲、闲置状态。
- 地图-终点修饰: 目的地的修饰词，用于更精确地描述终点，例如保定、南山区、广州塔。
- 地图-终点名称: 目的地的通用名称，例如家、老地方、郑州。
- 地图-终点目标: 目的地的具体目标，通常是专有名词，例如万达广场、全季酒店、汉庭酒店。
- 地图-终点类型: 目的地的类别，例如博物馆、超市、酒店。
- 地图-请求类型: 搜索请求的范围类型，例如四周、沿路、附近。
- 地图-起点修饰: 起点的修饰词，用于更精确地描述起点，例如苏州大学、金鸡湖。
- 地图-起点名称: 起点的通用名称，例如我现在的位置、莱阳、金鸡湖。
- 地图-起点类型: 起点的类别，例如加油站。
- 地图-距离: 搜索范围的距离，例如三公里以内、两公里内、五百米左右。
- 地图-距离排序: 搜索结果的排序方式，例如最近、最近的、离我最远。
- 地图-路线偏好: 导航路线的偏好设置，例如换成时间少的路段、走最便宜、高速优先。
- 地图-车载交互位置: 车内交互发生的位置，例如副驾。
- 地图-车载交互设备: 车内交互使用的设备，例如屏。
- 地图-途经点修饰: 途经点的修饰词，例如东城区、家、深圳北站。
- 地图-途经点名称: 途经点的通用名称，例如家庭地址、深圳北站、茂业百货。
- 地图-途经点目标: 途经点的具体目标，例如天安门。
- 地图-途经点类型: 途经点的类别，例如停车场、厕所、超市。
- 天气-__act__: 天气场景下的用户意图，例如确认、请求。
- 天气-__tgt__: 天气场景下用户意图的目标，例如护肤品、温度、运动。
- 天气-sys.指代: 系统定义的指代词，用于引用上文提到的城市。
- 天气-区域: 查询天气信息的行政区划，例如涞源县、静海县、鼓楼区。
- 天气-国家: 查询天气信息的国家，例如中国、塞舌尔、马拉维。
- 天气-地点: 查询天气信息的具体地点，例如朝阳陵园、武功山、苏南硕放机场。
- 天气-城市: 查询天气信息的城市，例如上海、北京、昆明。
- 天气-家务: 与天气相关的家务活动，例如擦玻璃、晒衣服。
- 天气-对象: 天气查询的对象，例如目的地。
- 天气-护肤品: 与天气相关的护肤品，例如防晒霜。
- 天气-日期: 查询天气的日期，例如三号、今天、明儿。
- 天气-时间: 查询天气的时间，例如一点、当前、早上。
- 天气-服装: 适宜当前天气的服装，例如大衣、棉大衣、衬衫。
- 天气-气象: 具体的气象现象，例如下雨不下、下雪、雾。
- 天气-活动: 与天气相关的活动，例如出去玩、洗车、逛公园。
- 天气-温差: 温度的差异，例如高。
- 天气-温度: 对温度的描述，例如冷、冷不冷啊、热不热。
- 天气-省份: 查询天气信息的省份，例如广东、江苏、辽宁。
- 天气-空气湿度: 空气的湿度情况，例如潮湿。
- 天气-空气质量: 空气的质量等级，例如好、最差。
- 天气-节日节气: 与天气查询相关的节日或节气，例如春节、端午节。
- 天气-装备: 应对天气所需的装备，例如伞、太阳伞、雨伞。
- 天气-运动: 适宜当前天气的运动，例如户外跑步、打球、爬山。
- 天气-阴历日期: 中国农历日期，例如农历正月二十二、正月二十五。
- 天气-风力: 风力的大小，例如个大、大、大吗。
- 影视-__act__: 影视场景下的用户意图，例如查询更多、请求。
- 影视-__tgt__: 影视场景下用户意图的目标，例如导演、演员、片名。
- 影视-sys.指代: 系统定义的指代词，用于引用上文提到的导演、演员、片名。
- 影视-上映时间: 影视作品的上映时间，例如一九九八年、昨天、最近。
- 影视-人数: 描述影视作品受欢迎程度的词语，例如最红、火爆、热门。
- 影视-作品标签: 影视作品的特殊标签，例如代表作、巅峰之作、第一部。
- 影视-倍速: 视频播放的速度，例如max、两倍速。
- 影视-制作公司: 影视作品的制作公司，例如上海唐人电影制作公司。
- 影视-制作成本: 影视作品的制作成本，例如低成本、投资最高。
- 影视-国家地区: 影视作品的出品国家或地区，例如台湾、泰国、香港。
- 影视-季数: 电视剧的季数，例如七、二、十一。
- 影视-对象: 用户意图所指的影视对象，例如片子、综艺、视频。
- 影视-导演: 影视作品的导演，例如吴宇森、张艺谋、徐峥。
- 影视-序列号: 在列表中的顺序，例如+1、下一、第三个。
- 影视-应用名称: 播放影视的应用名称，例如优酷视频、爱奇艺、腾讯视频。
- 影视-影视标签: 影视作品的内容标签，例如儿童、功夫、烧脑。
- 影视-影视类型: 影视作品的类型，例如动漫、动画、恐怖。
- 影视-操作: 对影视内容进行的操作，例如取消、我想看、播放。
- 影视-来源: 影视内容的来源列表，例如播放列表、播放历史、收藏列表。
- 影视-清晰度: 视频的清晰度，例如准高清、清晰度调低、清晰度调高。
- 影视-演员: 影视作品的演员，例如他和李沁、刘德华和梁朝伟、梁朝伟。
- 影视-片名: 影视作品的名称，例如从前有座灵剑山、天线宝宝、欧利亚。
- 影视-片长: 影视作品的时长，例如一个小时。
- 影视-电影人: 电影从业者，例如宋仲基、梁朝伟、邓超。
- 影视-电影公司: 电影制作发行公司，例如山影、漫威、迪士尼。
- 影视-电影奖: 电影奖项，例如奥斯卡、金马奖。
- 影视-票房: 电影的票房收入，例如过十亿。
- 影视-类似电影: 指代与某部电影相似的作品，例如这部电影。
- 影视-编剧: 影视作品的编剧，例如宁财神。
- 影视-视频源: 视频的来源，例如在线、本地、网络。
- 影视-视频结构: 视频的组成部分，例如片头、片尾曲。
- 影视-评分: 对影视作品的评价，例如asc（升序）、好看。
- 影视-语种: 影视作品的语言，例如中配、英文、英文版。
- 影视-车载交互位置: 车内交互发生的位置，例如主驾、前排、副驾。
- 影视-车载交互设备: 车内交互使用的设备，例如屏幕、显示屏。
- 影视-进度: 视频播放的进度，例如30分钟、十分钟、快进到1小时5分5秒。
- 影视-适用人群: 影视作品的适用人群，例如小朋友、情侣。
- 影视-适用年龄: 影视作品的适用年龄，例如三十岁、十岁。
- 影视-部数: 影视作品系列的数量，例如三、二。
- 影视-集数: 电视剧的集数，例如上集、六、第一集。
- 打电话-__act__: 打电话场景下的用户意图，例如请求。
- 打电话-__tgt__: 打电话场景下用户意图的目标，例如号码、联系人。
- 打电话-sys.序列号: 系统定义的列表顺序，例如#1、最后一个。
- 打电话-sys.页码: 系统定义的列表页码，例如下一页。
- 打电话-号码: 电话号码，例如134、13660216082、幺三八。
- 打电话-对象: 用户意图所指的对象，例如号、号码、电话。
- 打电话-归属地: 电话号码的归属地，例如上海、苏州。
- 打电话-操作: 用户的具体操作，例如呼叫、打、重拨。
- 打电话-电话储存信息: 手机中储存的电话相关信息，例如未接来电、联系人、通话记录。
- 打电话-电话标记: 联系人的备注或标签，例如工作、秘书。
- 打电话-电话类型: 电话的类型，例如座机、手机。
- 打电话-联系人: 电话联系人的姓名，例如LOVE、严焕红、郡主。
- 打电话-运营商: 电话号码所属的运营商，例如中国移动、移动、联通。
- 打电话-预置电话类型: 系统或服务预设的电话类型，例如售后、官方客服。
- 打电话-黄页号码: 通过黄页查询的机构或企业电话，例如奥凯航空客服、比亚迪客服电话。
- 播放控制-倍速: 媒体播放的速度，例如+、max、最小。
- 播放控制-对象: 播放控制所作用的对象，例如全屏观看、播放列表、播放历史。
- 播放控制-序列号: 播放列表中的顺序控制，例如+1、-1、下。
- 播放控制-播放模式: 媒体的播放模式，例如按次序播、挨个放、随机播放。
- 播放控制-操作: 对播放进行的操作，例如取消、打开、退出。
- 播放控制-进度: 控制播放的进度，例如1天、三十分钟、两分钟。
- 播放控制-音质: 播放的音质，例如标准音质。
- 收音机-对象: 收音机功能中的操作对象，例如收音机、电台、频道。
- 收音机-操作: 对收音机功能进行的操作，例如删了这个、播放、返回。
- 收音机-来源: 收音机频道的来源，例如播放列表、播放历史、收藏列表。
- 收音机-车载交互位置: 车内交互发生的位置，例如副驾驶。
- 收音机-车载交互设备: 车内交互使用的设备，例如屏幕。
- 收音机-频道: 收音机的频率或频道，例如104.3、一零一点零、九十二点五。
- 收音机-频道类型: 收音机的波段类型，例如调幅、调频、调频FM。
- 车载控制-action: 车载控制中的具体动作，例如调节。
- 车载控制-body: 车载控制（如座椅）相关的身体部位，例如屁股、背部。
- 车载控制-feature: 车载控制中的具体功能点，例如加热。
- 车载控制-object: 车载控制的对象，例如座椅。
- 车载控制-part: 车载控制功能中的可调节部分，例如温度。
- 车载控制-value: 车载控制的调节值或方向，例如吹脸、浓度调低一点、调小。
- 车载控制-位置: 车内控制所涉及的位置，例如前排、副驾、左前。
- 车载控制-功能: 车载系统的某项功能，例如壁纸桌面、手机无线充电、驻车。
- 车载控制-子功能: 某项功能下的子功能，例如前向碰撞预警、危险动作检测报警、车道偏向预警。
- 车载控制-对象: 车载控制的具体对象，例如空调、车内灯、阅读灯。
- 车载控制-对象功能: 控制对象所具备的功能，例如加热、按摩、混响。
- 车载控制-序列号: 在列表中的顺序，例如+1、上一个、下一个。
- 车载控制-座椅记忆位置: 座椅记忆的档位，例如1、二、副驾位。
- 车载控制-摄像头模式: 车载摄像头的工作模式，例如录音、照相延时、短视频拍摄。
- 车载控制-操作: 对车载功能进行的操作，例如关闭、设置、转到。
- 车载控制-操作_concrete: 辅助构成操作指令的词，例如true、为、成。
- 车载控制-方向偏移量: 调节的方向或幅度，例如max、最前、最后。
- 车载控制-模式: 车辆或系统的某种模式，例如影院模式、自动、舒享模式。
- 车载控制-调节内容: 需要调节的具体内容，例如模式、浓度、风向。
- 车载控制-身体位置: 与控制相关的身体部位，例如头部、肩部、脚部。
- 车载控制-车内灯类型: 车内灯光的具体类型，例如心跳氛围灯、氛围灯、阅读灯。
- 车载控制-车外灯类型: 车外灯光的具体类型，例如大灯、示宽灯、示廓灯。
- 车载控制-车机来源: 车机互联的来源，例如HUAWEI HICAR。
- 车载控制-车机模块: 车机系统的功能模块，例如多媒体、媒体、语音。
- 车载控制-音效: 车载系统提示或模拟的音效，例如借过提醒、拖拉机启动声、跑车发动机启动声。
- 车载控制-页面: 车机系统的界面或页面，例如设置、设置页面、配置。
- 音乐-__act__: 音乐场景下的用户意图，例如请求。
- 音乐-__tgt__: 音乐场景下用户意图的目标，例如专辑名、歌手名、歌曲名。
- 音乐-sys.指代: 系统定义的指代词，用于引用上文提到的专辑名、歌手名、歌曲名。
- 音乐-专辑名: 音乐专辑的名称，例如冬日浪漫、叶惠美、最伟大的作品。
- 音乐-主题: 音乐所表达或相关的主题，例如列车、母亲节、竞速小英雄。
- 音乐-主题曲类型: 歌曲作为主题曲的类型，例如主题曲、电影原声、配乐。
- 音乐-乐器: 歌曲中包含或与歌曲相关的乐器，例如二胡、架子鼓、钢琴。
- 音乐-作曲: 歌曲的作曲人，例如李宗盛、林俊杰、柳重言。
- 音乐-作词: 歌曲的作词人，例如方文山、李荣浩、林夕。
- 音乐-对象: 用户意图所指的音乐对象，例如歌、歌曲、音乐。
- 音乐-年代: 音乐作品所属的年代，例如七十年代、九十年代、八十年代。
- 音乐-年份: 音乐作品所属的年份，例如二零一零年、二零二一、二零二零年。
- 音乐-序列号: 列表中的顺序，例如下一个、十六。
- 音乐-应用名称: 播放音乐的应用，例如网易云音乐、美人鱼、酷我音乐。
- 音乐-排行榜: 音乐的排行榜单，例如原创榜、热歌排行榜、热门翻唱榜。
- 音乐-播放列表: 用户创建或收藏的歌单，例如闹钟。
- 音乐-操作: 对音乐进行的操作，例如打开、推荐、播放。
- 音乐-日期: 与音乐相关的特定日期，例如20250613。
- 音乐-时间: 与音乐相关的特定时间，例如11:12:31。
- 音乐-歌手名: 演唱歌曲的歌手，例如周杰伦、杨宗纬、阿杜。
- 音乐-歌手性别: 歌手的性别，例如男、男生。
- 音乐-歌曲名: 歌曲的名称，例如我是如此的相信、橘子之歌。
- 音乐-歌曲结构: 歌曲的组成部分，例如副歌、高潮。
- 音乐-民族: 与音乐相关的民族，例如藏族。
- 音乐-版本: 歌曲的不同版本，例如DJ、改编、现场。
- 音乐-语种: 歌曲的语言，例如中国、粤语、英文。
- 音乐-车载交互位置: 车内交互发生的位置，例如二排、二排左、副驾驶。
- 音乐-车载交互设备: 车内交互使用的设备，例如屏、屏幕、顶部娱乐屏。
- 音乐-进度: 音乐播放的进度，例如五分钟。
- 音乐-适用人群: 音乐的适用人群，例如宝宝、老人、老年人。
- 音乐-适用年龄: 音乐的适用年龄，例如一岁、五岁、四岁。
- 音乐-重复次数: 音乐播放的重复次数，例如一。
- 音乐-音乐场景: 适合播放音乐的场景，例如看书、起床、运动。
- 音乐-音乐类型: 音乐的类型，例如催眠曲、金属、黑人音乐。
- 音乐-音乐风格: 音乐的风格，例如忧郁、欢快、热血。
- 音乐-音源: 音乐的来源，例如优盘、手机、蓝牙。
- 音乐-音质: 音乐的音质标准，例如杜比音乐。
"""


# --- 【核心修改】合并后的系统提示 ---
# 指导LLM同时执行意图识别和槽位填充，并输出统一的列表格式
SYSTEM_PROMPT_TEMPLATE = f"""
你是一个专业的车载系统自然语言理解（NLU）专家。
你的任务是基于用户的查询（Query），同时完成两项任务：
1.  **意图识别 (Intent Classification)**: 识别出查询中包含的所有领域（Domain）和意图（Intent）。
2.  **槽位填充 (Slot Filling)**: 抽取出与每个意图相关的槽位（Slot）和槽位值（Value）。

你需要严格遵循以下规则：
1.  **识别多个语义帧**: 用户的单次查询可能包含多个独立的意图。你需要为每一个意图生成一个对应的语义结构。
2.  **严格匹配**: 识别出的领域、意图和槽位名必须严格从下面的列表中选择。不允许创造任何列表中不存在的名称。
3.  **关联性**: 提取的槽位必须与识别出的意图直接相关。
4.  **输出格式**: 你的输出必须是一个严格的JSON **List** (列表)。列表中的每一个JSON对象代表一个完整的语义帧，包含领域、意图和对应的槽位。
    - **单个意图的格式**: `[{{"domain": "领域", "intent": "意图", "slots": {{"槽位名1": "槽位值1", "槽位名2": "槽位值2"}}}}]`
    - **多个意图的格式**: `[{{"domain": "领域1", "intent": "意图1", "slots": {{...}}}}, {{"domain": "领域2", "intent": "意图2", "slots": {{...}}}}]`
    - **没有槽位的情况**: 如果某个意图没有关联的槽位，`slots`字段应为一个空对象: `{{"domain": "领域", "intent": "意图", "slots": {{}}}}`
5.  **空结果**: 如果用户的查询没有匹配到任何领域和意图，请返回一个空的列表: `[]`
6.  **不要包含任何解释**: 你的最终回答中，除了要求的JSON列表，不要包含任何其他文字、解释或注释。
7.  **必要知识**: 对于“车载控制”领域，“车机控制”针对屏幕等虚拟对象（如“连接蓝牙”）；“车身控制”针对车窗、空调等物理实体（如“空调温度调低一点”）；“提供信息”用于无实体的情况。

---
**可选的领域和意图列表：**
{DOMAIN_INTENT_LIST}
---
**可选的槽位列表：**
{SLOT_LIST}
---

"""

def setup_arg_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description="使用 LLM API 进行统一的语音语言理解 (SLU)")
    parser.add_argument(
        "--input-file", type=str, required=True, help="输入的JSONL元数据文件路径"
    )
    parser.add_argument(
        "--audio-dir", type=str, required=True, help="存放音频文件 (.wav) 的目录路径"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="输出的JSONL文件路径"
    )
    
    # --- 修改 ---: 增加'local'选项，并更新help说明
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "azure", "local"],
        help="API提供商: 'google'/'azure' (通过Dashscope), 或 'local' (本地OpenAI兼容接口)"
    )
    # --- 新增 ---: 为本地部署模型增加 --api-base 参数
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://0.0.0.0:12355/v1",
        help="本地LLM服务的API基地址 (仅当 --provider='local' 时使用)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM服务的API key。本地服务通常不需要，云服务则必需。"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash",
        help="要使用的模型名称 (例如: 'gemini-2.5-flash', 或本地部署的模型名)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="生成文本的温度。"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="模型生成的最大token数量。"
    )
    return parser

def encode_audio_to_base64(audio_path: Path) -> Optional[str]:
    """读取音频文件，进行Base64编码，并返回字符串。"""
    try:
        with open(audio_path, "rb") as audio_file:
            binary_data = audio_file.read()
            return base64.b64encode(binary_data).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"音频文件未找到: {audio_path}")
        return None
    except Exception as e:
        logging.error(f"编码音频文件时出错 {audio_path}: {e}", exc_info=True)
        return None

# --- 新增 ---: 专门用于调用本地 OpenAI 兼容接口的函数
def call_local_api(
    api_base: str,
    model_name: str,
    audio_path: Path,
    temperature: float,
    max_tokens: int
) -> Optional[str]:
    """
    使用 openai 库向本地部署的 OpenAI 兼容服务发送 SLU 请求。
    """
    if OpenAI is None:
        raise ImportError("要使用本地提供商，请先安装openai库: pip install openai")

    client = OpenAI(base_url=api_base, api_key="not-needed")
    
    audio_base64 = encode_audio_to_base64(audio_path)
    if not audio_base64:
        return None

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
                        },
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"调用本地 API 时发生错误 (audio: {audio_path.name}): {e}", exc_info=True)
        return None



import argparse
import json
import logging
import base64
import os
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any

# --- 新增 ---: 导入 OpenAI 库用于本地接口调用
try:
    from openai import OpenAI
except ImportError:
    # 如果用户不使用本地模式，这个库不是必需的
    OpenAI = None 

# 使用 tqdm 显示进度条
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        print("tqdm is not installed, progress bar will not be shown. "
              "Install it with: pip install tqdm")
        return iterable

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def setup_arg_parser() -> argparse.ArgumentParser:
    # ... [此函数未修改，省略] ...
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description="使用 LLM API 进行统一的语音语言理解 (SLU)")
    parser.add_argument(
        "--input-file", type=str, required=True, help="输入的JSONL元数据文件路径"
    )
    parser.add_argument(
        "--audio-dir", type=str, required=True, help="存放音频文件 (.wav) 的目录路径"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="输出的JSONL文件路径"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["google", "azure", "local"],
        help="API提供商: 'google'/'azure' (通过Dashscope), 或 'local' (本地OpenAI兼容接口)"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://0.0.0.0:12355/v1",
        help="本地LLM服务的API基地址 (仅当 --provider='local' 时使用)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM服务的API key。本地服务通常不需要，云服务则必需。"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash",
        help="要使用的模型名称 (例如: 'gemini-2.5-flash', 或本地部署的模型名)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="生成文本的温度。"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="模型生成的最大token数量。"
    )
    return parser

def encode_audio_to_base64(audio_path: Path) -> Optional[str]:
    # ... [此函数未修改，省略] ...
    """读取音频文件，进行Base64编码，并返回字符串。"""
    try:
        with open(audio_path, "rb") as audio_file:
            binary_data = audio_file.read()
            return base64.b64encode(binary_data).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"音频文件未找到: {audio_path}")
        return None
    except Exception as e:
        logging.error(f"编码音频文件时出错 {audio_path}: {e}", exc_info=True)
        return None

def call_local_api(
    api_base: str,
    model_name: str,
    audio_path: Path,
    temperature: float,
    max_tokens: int
) -> Optional[str]:
    # ... [此函数未修改，省略] ...
    """
    使用 openai 库向本地部署的 OpenAI 兼容服务发送 SLU 请求。
    """
    if OpenAI is None:
        raise ImportError("要使用本地提供商，请先安装openai库: pip install openai")

    client = OpenAI(base_url=api_base, api_key="not-needed")
    
    audio_base64 = encode_audio_to_base64(audio_path)
    if not audio_base64:
        return None

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
                        },
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"调用本地 API 时发生错误 (audio: {audio_path.name}): {e}", exc_info=True)
        return None

# --- 核心修复 ---
# 对 call_cloud_api 函数进行修改，以支持 provider-specific 的请求体
def call_cloud_api(
    api_key: str,
    provider: str,
    model_name: str,
    audio_path: Path,
    temperature: float,
    max_tokens: int
) -> Optional[str]:
    """
    使用 requests 库向云服务（通过Dashscope）发送SLU请求。
    此版本修复了对不同 provider（google vs azure）的请求体差异。
    """
    API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    audio_base64 = encode_audio_to_base64(audio_path)
    if not audio_base64:
        return None

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    user_content = []
    # --- 修复点 ---: 根据 provider 构建不同的 user_content
    if provider == "google":
        dashscope_file_url = f"data:audio/wav;base64,{audio_base64}"
        user_content.append({"type": "audio_url", "audio_url": {"url": dashscope_file_url}})
    elif provider == "azure":
        # Azure GPT-4o 音频模型需要不同的格式
        user_content.append({"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}})
    else:
        logging.error(f"不支持的云提供商: {provider}")
        return None

    # 可以选择性地添加一个文本部分，引导模型专注于SLU任务
    user_content.append({"type": "text", "text": "请根据系统指令处理提供的音频。"})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": user_content}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        'dashscope_extend_params': {'provider': provider}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        # --- 调试点：检查响应中是否有错误字段 ---
        if 'error' in response_json:
            logging.error(f"API返回了错误信息: {response_json['error']}")
            return None # 如果有错误，直接返回None

        return response_json['choices'][0]['message']['content'].strip()

    except requests.exceptions.RequestException as e:
        logging.error(f"调用云 API 时发生网络错误 (audio: {audio_path.name}): {e}")
        # --- 调试点：打印出详细的响应体 ---
        if e.response is not None:
            logging.error(f"API 响应状态码: {e.response.status_code}")
            logging.error(f"API 响应内容: {e.response.text}") # 这一行最关键
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"解析云 API 响应时出错: {e}. Response: {response.text}")
        return None
def process_file(args: argparse.Namespace):
    """
    读取元数据文件，加载对应音频，根据provider选择API处理，并将结果写入输出文件。
    """
    input_file, audio_dir, output_file = Path(args.input_file), Path(args.audio_dir), Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"错误: 输入文件未找到 {input_file}")
        return

    logging.info(f"开始处理文件 {input_file}，共 {len(lines)} 条记录... 使用 provider: {args.provider}")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(lines, desc="处理音频中"):
            try:
                data = json.loads(line)
                item_id = data.get("id")
                ground_truth_query = data.get("query")
                if not item_id:
                    logging.warning(f"跳过格式错误的行（缺少id）: {line.strip()}")
                    continue
                
                audio_path = audio_dir / f"id_{item_id}.wav"
                if not audio_path.exists():
                    logging.warning(f"跳过记录，找不到音频文件: {audio_path}")
                    continue

                model_output_str = None
                # --- 修改 ---: 根据 provider 选择不同的 API 调用函数
                if args.provider == "local":
                    model_output_str = call_local_api(
                        api_base=args.api_base,
                        model_name=args.model_name,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens
                    )
                else: # google or azure
                    model_output_str = call_cloud_api(
                        api_key=args.api_key,
                        provider=args.provider,
                        model_name=args.model_name,
                        audio_path=audio_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens
                    )

                # 解析逻辑保持不变
                parsed_semantics_list: List[Dict[str, Any]] = []
                if model_output_str:
                    try:
                        if model_output_str.startswith("```json"):
                            model_output_str = model_output_str.strip("```json\n").strip("`")
                        
                        parsed_output = json.loads(model_output_str)

                        if isinstance(parsed_output, list):
                            parsed_semantics_list = parsed_output
                        else:
                            raise json.JSONDecodeError("Model output is not a list", model_output_str, 0)

                    except json.JSONDecodeError as e:
                        logging.warning(f"\n无法解码 JSON。ID: {item_id}, Error: {e}, Output: {model_output_str}")
                else:
                    logging.warning(f"\nAPI 调用失败或返回空。ID: {item_id}。")

                result = {"id": item_id, "query": ground_truth_query, "semantics": parsed_semantics_list}
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

            except Exception as e:
                logging.error(f"处理行时发生意外错误: {line.strip()}. 错误: {e}", exc_info=True)
            
    logging.info(f"\n处理完成。结果已保存到 {output_file}")


def main():
    """主函数"""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # --- 修改 ---: 仅在非本地模式下检查 API Key
    if args.provider != "local":
        if not args.api_key:
            args.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not args.api_key:
            logging.error(f"错误: 使用 '{args.provider}' 提供商时，必须提供 API key。")
            return
    
    if args.provider == "local" and OpenAI is None:
        logging.error("错误: 要使用 'local' 提供商, 请先安装 openai 库: pip install openai")
        return

    process_file(args)

if __name__ == "__main__":
    main()