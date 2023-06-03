from dotenv import load_dotenv
import openai, os, requests

load_dotenv()
openai.api_key = os.getenv('APIKEY')

def get_completion(prompt, model="gpt-3.5-turbo"):
    message = [
        {"role": "user",
        "content": prompt
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=message,
        temperature=0,
    )
    return response.choices[0].message["content"]

# def get_public_ip():
#     try:
#         response = requests.get('https://api.ipify.org?format=json')
#         ip_info = response.json()
#         public_ip = ip_info['ip']
#         return public_ip
#     except requests.RequestException:
#         return 'Error: Failed to retrieve public IP'

# current_ip = get_public_ip()
# print('Current public IP:', current_ip)

text = f"""
    因变量y：因变量是机器学习模型的输出，也就是机器学习模型的预测值。
    - 特征x：特征是机器学习模型的输入，也就是机器学习模型的输入项。特征可以有很多个，比如说房价预测模型的特征有房屋面积、房间数量、卧室数量、卫生间数量、楼层数量等等。如果有n个这样的特征，那么每一个数据点（每一个样本）就可以表示为一个n维的向量。
    - 特征工程：其实就是对原始数据进行预处理，以便建模的一个阶段，主要包括特征提取、特征选择、特征变换、特征降维等等。其主要目的是创建出对于问题来说最有意义的特征组合，以提升模型输出的准确性。除了特征缩放，异常值处理、缺失值处理、特征编码这些基本的预处理步骤之外，特征工程还需要理解特定专业领域的知识，才能准确的建模。
        - 特征缩放：预处理数据时常用的步骤，主要原因是很多特征的尺度和单位差异很大，其目的是在于确保所有特征在数值上处于同一数量级，梯度下降如果没有正确的进行特征缩放，可能无法收敛。
        - 归一化：将特征缩放到0到1之间，公式：(x - min) / (max - min)
        - 标准化：将特征缩放到均值为0，方差为1的正态分布中，公式：(x - mean) / std  
    - 模型权重参数w：模型权重参数是机器学习模型的参数，其维度数量与特征的维度数量相同，比如说房价预测模型的特征有房屋面积、房间数量、卧室数量、卫生间数量、楼层数量等等，那么模型权重参数就有5个，分别是w1、w2、w3、w4、w5。
    - 偏置参数b：偏置参数是机器学习模型的参数，其维度数量为1，比如说房价预测模型的偏置参数就是b，可以视为一个基准的房价，所有的房价都是在这个基准房价上进行调整的。
    - 向量化：为了对大量数据进行科学计算和分析，使用向量化的编程技巧来避免显式循环，从而提高计算效率，加快训练。比如利用numpy的数组进行元素级别的操作，直接把两个数组相加，让第三个数组的每个元素等于前两个数组对应元素的和，而不是使用for循环来实现。
    """
prompt = f"""
    把以上内容总结成一句话，并用三个反引号包裹。
    ```{text}```
    """
response = get_completion(prompt)
print(response)
