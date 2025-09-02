# Neuron-Musk

## 소개

> 뇌 신호 데이터(trace)에서 노이즈와 스파이크를 효과적으로 분류하고, 추출된 스파이크의 특징을 분석하여 클러스터링하는 딥러닝 기반의 파이프라인입니다.
> 

**문제 정의**

가느다란 전극을 뇌조직에 삽입하면, 주변 뉴런이 전기신호를 보낼 때마다 μV 단위의 전압 변동이 기록됩니다. 이때 입력되는 신호에는 노이즈와 다양한 뉴런들의 전기신호가 섞여 들어오기 때문에 정확한 분류가 필요합니다.

또한 서로 다른 종류의 뉴런들(예: 교감신경 vs 비교감신경)은 각각 다른 진폭의 신호를 보인다고 알려져 있습니다.

따라서 입력된 신호를 먼저 <노이즈/스파이크>로 분류하고, 이후 스파이크들을 <뉴런#1, 뉴런#2...> 등으로 구분하는 모델을 만들고자 합니다.

![위와 같은 파이프라인으로 Spike Sorting은 진행됩니다.](https://img.notionusercontent.com/s3/prod-files-secure%2Fcaac11a1-578d-4638-bf54-1d47cd3de8ed%2Fac846088-ac7a-4151-99c0-12776fb4216a%2Fimage.png/size/w=2000?exp=1756923799&sig=ykyCJPcM7E_CoHDHUzrRYPRZONr8EMN89xtY8LRYqeI&id=262a7930-e09c-806e-8213-ebd7cf626f3c&table=block)

위와 같은 파이프라인으로 Spike Sorting은 진행됩니다.

## 목표

### 최종 목표

다양한 종류의 뉴런 신호가 섞인 복잡한 데이터(trace) 속에서 특정 뉴런의 발화 신호(Spike)를 정확하게 탐지하고, 각 스파이크가 어떤 뉴런에서 발생했는지 구분해내는 자동화된 모델을 개발합니다.

### 세부 목표

- **스파이크 탐지 (Spike Detection)**
    - 배경 노이즈(Noise)와 실제 신경 발화 신호(Spike)를 높은 정확도로 분류하는 딥러닝 모델을 구축합니다.
    - Wavelet Transform과 1D CNN을 활용해 신호의 미세한 특징을 학습하고, 이를 통해 분류 성능을 극대화합니다.
- **특징 추출 및 클러스터링 (Feature Extraction & Clustering)**
    - 탐지된 각 스파이크 신호로부터 고유한 특징(Embedding)을 추출합니다.
    - 추출된 특징을 기반으로 비슷한 형태의 스파이크들을 그룹화(Clustering)하여 개별 뉴런(#1, #2, ...)을 식별합니다.
    - S4(SSM) 모델과 HDBSCAN 같은 기법을 활용해 클러스터링의 정확도를 높입니다.

## 데이터셋

| **데이터셋** | **설명** |
| --- | --- |
| Spikeforest
https://spikeforest.flatironinstitute.org/ | spikeforest란 전세계의 많은 연구실에서 직접 측정한 데이터, 시뮬레이션을 통해 합성한 데이터 등 많은 종류의 데이터들과 만든 모델을 돌려 나온 결과값들을 비교해볼 수 있는 웹사이트입니다.
많은 데이터 중에서 SYNTH_MONOTRODE라는 데이터를 사용하였으며, 이는 시뮬레이션을 통해 합성된 싱글 채널 데이터입니다.
SYNTH_MONOTRODE라는 데이터에는 아래 그림과 같은 데이터가 총 110개 존재하고 있습니다. 모두 길이는 144,000이지만 각각의 데이터마다 spike의 위치와 종류는 다릅니다.
각 데이터에는 trace 뿐 아니라 그 데이터 내에서 몇 개의 뉴런이 발화하였는지, spike의 프레임, 종류가 무엇인지와 같은 데이터가 들어있습니다. |
|  |  |

![해당 데이터에 대해 다른 모델들의 accuracy입니다. 상기 모델들이 quiroga_easy 데이터(노이즈가 적은 데이터)에 대해서는 98%, 97%같은 높은 수치를 보여주고 있습니다. 상대적으러 노이즈가 큰 데이터에 대해서는 최대 80%정도의 accuracy를 보여주고 있습니다](https://file.notion.so/f/f/caac11a1-578d-4638-bf54-1d47cd3de8ed/3d5c568e-e2d7-4099-b6a7-5b6724af625f/image.png?table=block&id=262a7930-e09c-80c3-b50c-c418d51df20e&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&expirationTimestamp=1756864800000&signature=GH6PE5-FFI037wAQagxR-tsI6K0ee-u9C1siPDCPzU8&downloadName=image.png)

해당 데이터에 대해 다른 모델들의 accuracy입니다. 상기 모델들이 quiroga_easy 데이터(노이즈가 적은 데이터)에 대해서는 98%, 97%같은 높은 수치를 보여주고 있습니다. 상대적으러 노이즈가 큰 데이터에 대해서는 최대 80%정도의 accuracy를 보여주고 있습니다

![SYNTH_MONOTRODE 데이터 중 하나, (144,000,)의 shape을 가지며, 3 개의 다른 뉴런이 발화한 데이터](https://file.notion.so/f/f/caac11a1-578d-4638-bf54-1d47cd3de8ed/7e5b4550-4cab-4027-a512-0e2798ef280a/image.png?table=block&id=262a7930-e09c-80a0-b2c7-d1648ddce85a&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&expirationTimestamp=1756864800000&signature=9KhvZihfmlRUtrf0yFSVVKCVR3B3laIPTkRl54pP88Q&downloadName=image.png)

SYNTH_MONOTRODE 데이터 중 하나, (144,000,)의 shape을 가지며, 3 개의 다른 뉴런이 발화한 데이터

![위 전체 trace 중 일부(맨위), 300~6000hZ의 주파수만 걸러낸 신호(중간), 나머지 대역의 신호(맨 하단)](https://aiku.notion.site/image/attachment%3A3664bd37-0cc3-46d4-ae88-986cada3c344%3Aimage.png?table=block&id=262a7930-e09c-808f-8525-d210a0a3576a&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=1010&userId=&cache=v2)

위 전체 trace 중 일부(맨위), 300~6000hZ의 주파수만 걸러낸 신호(중간), 나머지 대역의 신호(맨 하단)

![spike만을 추출한 waveform의 일부입니다.]([attachment:1d593aa8-6d52-4065-a022-53223c22ca73:image.png](https://aiku.notion.site/image/attachment%3A1d593aa8-6d52-4065-a022-53223c22ca73%3Aimage.png?table=block&id=262a7930-e09c-80cf-a38d-fee56e0a0037&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=460&userId=&cache=v2))

| ![Spike](https://aiku.notion.site/image/attachment%3A1d593aa8-6d52-4065-a022-53223c22ca73%3Aimage.png?table=block&id=262a7930-e09c-80cf-a38d-fee56e0a0037&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=460&userId=&cache=v2) | ![Noise](https://aiku.notion.site/image/attachment%3A33eb18f7-dd60-4306-859e-efe4c68c0b9f%3Aimage.png?table=block&id=262a7930-e09c-8018-9cda-e74252b5454e&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=460&userId=&cache=v2) |
|----------------------------------------|-----------------------------------------|
| spike만 추출한 waveform                | noise를 추출한 waveform                 |

| ![Mixed1](https://aiku.notion.site/image/attachment%3Aa0811e94-d5a6-466c-9046-742adabd432b%3Aimage.png?table=block&id=262a7930-e09c-8094-a406-df5c0a4d96dc&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=460&userId=&cache=v2) | ![Mixed2](https://aiku.notion.site/image/attachment%3Aaadad27d-cd74-4ca9-91ec-ce9910f79c50%3Aimage.png?table=block&id=262a7930-e09c-80cf-a042-cb5306019079&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=460&userId=&cache=v2) |
|------------------------------------------|-------------------------------------------|
| 혼합 데이터 예시 1                       | 혼합 데이터 예시 2                        |

## 모델링

이 프로젝트는 크게 Spike detection과 feature extraction & Clustering 두 가지로 나눌 수 있습니다.

그래서 Spike detection과 feature extraction&clustering 각각 다르게 모델링을 진행하였습니다.

### Spike Detection

- trace가 들어오게 되면 threshold 방식으로 먼저 스파이크를 일차적으로 걸러주게 됩니다.
- 들어온 spike를 wavelet transform을 진행하게 됩니다. 
저주파부터 고주파 성분까지 총 3개의 계수 리스트가 생성됩니다.
- 각 리스트에 대하여 1D CNN을 진행합니다.
    
![Spike Detection Pipeline](https://aiku.notion.site/image/attachment%3A17473bda-d96b-4e0c-b885-2eddf10543ec%3Aa63cf81e-258f-4bdf-a480-5885a6c1d22a.png?table=block&id=262a7930-e09c-80a6-a377-d0ef63adaa7b&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=950&userId=&cache=v2)
    
    Spike Detection Pipeline
    

### Feature Extraction & Clustering

![image.png](https://aiku.notion.site/image/attachment%3A2ed8fb6a-957a-417b-be8e-f303e0384c69%3Aimage.png?table=block&id=262a7930-e09c-800e-ac77-ce3316b36afe&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=1010&userId=&cache=v2)

| **단계** | **Input** | **Output** |
| --- | --- | --- |
| **Spike Detection & Waveform Extraction** | 원시 시계열 데이터 | waves, timestamps |
| **S4 Encoder 학습** | waves, labels(optional) | 학습된 모델, Z(임베딩) |
| **Clustering** | Z | pred, metrics |
| **Evaluation** | pred, labels | accuracy, ARI, NMI |
| **Visualization** | Z, labels | spike_embedding_tsne.png |
- SpikeInterface로 얻은 spike window를 **S4(SSM)** 인코더로 임베딩하고, HDBSCAN로 뉴런별 clustering한 뒤, ARI/NMI/Accuracy로 품질 평가까지 한 번에 수행.
- GT가 없으면 라벨을 -99로 채워 평가 안전장치를 두었고, 라벨이 있을 때 SupCon로 성능을 더 끌어올리는 구조.

### GT 라벨 생성

- 피크 샘플 인덱스와 GT 스파이크 시점을±tol_ms 안에서 가장 가까운 것으로 붙여 뉴런 ID 라벨을 만든다.
    - 없으면 -99로 채움(평가에서 제외되도록 하기 위함). GT 있으나 매칭 실패는 -1로 남겨둔다. (원한다면 평가에서 -1도 제외 마스크로 처리).
- 현재 코드는  ****compute_cluster_accuracy()에서 -99만 제외하고 -1은 제외하지 않음.

### Loss

- InfoNCE 손실
    - 각 샘플에 대해 두 개의 증강된 뷰를 만들어 contrastive 학습에 사용. 같은 샘플의 두 증강 뷰(p1, p2)가 가까워지고, 다른 샘플과는 멀어지도록 학습.
    - 라벨(labels)이 있으면 supervised contrastive(SupCon) 학습을 함께 수행할 수 있음.
- supervised contrastive loss
    - 같은 클래스의 모든 샘플을 positive로 간주하여 더 풍부한 positive 관계를 학습 : 같은 뉴런의 spike waveforms가 feature 공간에서 더 밀집되도록 학습되어, 일반화 성능 향상.

> loss = loss + τ * supcon_loss(z, labels)
> 
- 라벨이 없는 경우 : InfoNCE만 사용 → 순수 self-supervised
- 라벨이 있는 경우 : InfoNCE + SupCon 혼합

## 훈련 및 평가

(각 모델을 훈련한 환경을 나열하고 평가 지표와 그 값을 서술하세요)

### Spike Detection

**노이즈와 spike를 모두 합하여 564,084개의 waveform에 대하여 wavelet transform을 진행했고 8:2로 나누어 학습을 진행하였습니다.**

- BCE Loss, Adam, 에폭 수 8

Epoch 1/8, Loss: 0.2513, Test Acc: 0.9137

Epoch 2/8, Loss: 0.2075, Test Acc: 0.9181

Epoch 3/8, Loss: 0.1990, Test Acc: 0.9176

Epoch 4/8, Loss: 0.1949, Test Acc: 0.9197

Epoch 5/8, Loss: 0.1918, Test Acc: 0.9211

Epoch 6/8, Loss: 0.1898, Test Acc: 0.9209

Epoch 7/8, Loss: 0.1877, Test Acc: 0.9219

Epoch 8/8, Loss: 0.1860, Test Acc: 0.9235

**Accuracy = 0.9235, Precision = 0.9255**

### Feature Extraction & Clustering

**총 6444개의 waveform에 대하여 학습을 진행하였고, 11194개의 waveform에 대하여 테스트를 진행하였습니다.**

[epoch 01] loss=8.4440

[epoch 02] loss=8.6010

[epoch 03] loss=8.4435

[epoch 04] loss=9.0735

[epoch 05] loss=8.1072

[epoch 06] loss=7.1415

[epoch 07] loss=6.4022

[epoch 08] loss=6.5096

[epoch 09] loss=6.6290

[epoch 10] loss=6.3136

[epoch 11] loss=6.5737

[epoch 12] loss=6.0055

[epoch 13] loss=6.7682

[epoch 14] loss=6.0323

[epoch 15] loss=5.8371

[epoch 16] loss=5.1776

[epoch 17] loss=4.9795

[epoch 18] loss=5.0266

[epoch 19] loss=4.8911

[epoch 20] loss=5.2379

[epoch 21] loss=4.9260

[epoch 22] loss=4.6464

[epoch 23] loss=4.3462

[epoch 24] loss=4.4797

[epoch 25] loss=4.6215

[epoch 26] loss=4.6868

[epoch 27] loss=4.3982

[epoch 28] loss=4.5831

[epoch 29] loss=4.3171

[epoch 30] loss=4.5996

[Cluster] metrics: {'ARI': 0.7857059890384587, 'NMI': 0.8221826133994493}

[Cluster] Accuracy: 0.8867

[Summary] clusters (id:count) — first 12: {-1: 1051, 0: 202, 1: 79, 2: 111, 3: 80, 4: 46, 5: 162, 6: 58, 7: 49, 8: 110, 9: 62, 10: 47}

## 결과

### Spike Detection

![image.png](https://aiku.notion.site/image/attachment%3Ac26bbc61-97fc-4362-8197-cd8d21c31cc8%3Aimage.png?table=block&id=262a7930-e09c-804b-904d-e9ff6de2c8cc&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=690&userId=&cache=v2)

노이즈가 어느 정도 들어가는 것보다 Spike가 노이즈로 분류되지 않도록 하는 것이 더 중요한데, Precision이 0.9255로 약 8%의 spike가 노이즈로 분류되었습니다.

### Feature Extraction & Clustering

![accuracy 90%](https://aiku.notion.site/image/attachment%3A82830cd2-1c29-49e4-88a0-39741a1ffa77%3AKakaoTalk_20250901_135546462.png?table=block&id=262a7930-e09c-8062-ab53-ce7f97cfb87f&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=1010&userId=&cache=v2)

accuracy 90%

![accuracy 80%](https://aiku.notion.site/image/attachment%3Ae4ee075f-1441-4dd1-89ba-246b0dc34dc2%3AKakaoTalk_20250901_135554038.png?table=block&id=262a7930-e09c-8066-b693-f2eb0a4c6c7e&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=1010&userId=&cache=v2)

accuracy 80%

![image.png](https://aiku.notion.site/image/attachment%3A338bae37-944f-4014-8a6c-797dd38bc0f7%3Aimage.png?table=block&id=262a7930-e09c-8065-aeb7-ca56a1f34cda&spaceId=caac11a1-578d-4638-bf54-1d47cd3de8ed&width=1010&userId=&cache=v2)

## 한계 분석

### Spike Detection

1. **동적 Threshold 방식의 불안정성**
- **문제점**: 실시간으로 변하는 신호(trace)의 중앙값(median)을 기반으로 스파이크를 탐지하는 threshold 방식은, 신호의 국소적인 변화에 민감하여 threshold 값이 불안정해집니다.

1. **Wavelet Transform의 정보 손실 문제**
- **문제점**: 24000Hz 샘플링 속도에서 36프레임 길이의 스파이크 파형에 Wavelet Transform을 적용했을 때, 유의미한 계수가 소수(2개)만 추출되었습니다.

### Feature Extraction & Clustering

1. **클러스터링 알고리즘의 한계**
- **문제점**: HDBSCAN은 밀도 기반 클러스터링으로 노이즈 처리에 강점이 있지만, **서로 다른 밀도를 가진 클러스터들이 인접해 있을 경우** 이들을 하나의 클러스터로 병합하거나 경계 부분을 노이즈로 처리할 수 있습니다.
1. **임베딩 공간에서의 클러스터 중첩 및 모호성**
- **문제점**: Spike Embedding 시각화 결과를 보면, pre-trained 모델(정확도 ~90%)이 original 모델(~80%)보다 개선되었음에도 불구하고 여전히 **여러 클러스터가 서로 겹치거나 가깝게 분포**하는 것을 확인할 수 있습니다.

### 전체적인 한계점

1. **객관적인 성능 비교의 부재**
- **문제점**: 프로젝트의 '한계 분석' 부분에서 언급되었듯이, 시간 관계상 **다른 최신 스파이크 분류(Spike Sorting) 모델과의 성능 비교를 수행하지 못했습니다.**
1. **일반화 성능에 대한 의문**
- **문제점**: 현재 모델은 `SYNTH_MONOTRODE` 라는 특정 데이터셋에 맞춰 학습 및 테스트되었습니다.

## 후속 프로젝트

1. 비교할 만한 모델을 찾아서 성능을 제대로 비교
2. Spike Detection 모델과 Clustering 모델을 합치기
3. SSM을 주력으로 하는 end-to-end 모델을 개발
