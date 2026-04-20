# MultimodalText 모델 계획서

## 문제 정의

이 계획서의 목표는 **텍스트 중심 수학 추론을 먼저 확실히 만들고**, 그 위에 **텍스트가 많은 이미지와 도식**을 받아들일 수 있는 멀티모달 인터페이스를 얹는 것이다. 중요한 전제는 OpenMythos가 entity["company","Anthropic","ai company"]의 실제 내부 구조 공개본이 아니라, 공개 문헌과 추정을 바탕으로 만든 **독립적·이론적 재구성**이라는 점이다. 따라서 아래 설계는 “상용 비공개 모델의 복제”가 아니라, **공개 연구 결과와 제공된 도식이 일관되게 가리키는 RDT 계열 연구용 설계안**으로 읽는 것이 맞다. citeturn1view0turn18view0

당신과의 앞선 대화 맥락을 반영하면, 이 프로젝트의 최우선 과제는 범용 VLM을 한 번에 만드는 것이 아니라 **작은 규모에서 수학적 절차를 안정적으로 학습하는 텍스트 코어**를 확보하는 일이다. 이 판단은 타당하다. entity["organization","DeepMind","ai lab"] Mathematics Dataset는 모듈별로 절차적으로 생성되는 Q/A 벤치마크이며, 모듈당 약 200만 개의 학습 예제를 공개했고, 학습은 `train-easy`, `train-medium`, `train-hard`로 나뉘며, 질문은 160자, 답은 30자로 제한된다. 즉 초기 병목은 긴 문맥 기억보다 **숫자 위치 추적, 중간 상태 보존, 모듈 조합 추론**에 가깝다. citeturn1view2turn12view0turn10search0

이 점 때문에 MultimodalText의 1차 목표는 “거대한 멀티모달 범용 모델”이 아니라, **짧은 텍스트·짧은 답·정확한 알고리즘적 추론**에 최적화된 작은 반복형 코어를 만들고, 그 뒤에 **문서·스크린샷·수식 이미지**를 위한 시각 입력 파이프라인을 얹는 것이다. 이 순서는 최근 반복 깊이 모델 연구의 핵심 주장과도 맞는다. 최근 연구들은 많은 추론 과제가 파라미터 수 그 자체보다 **유효 깊이**를 더 요구하며, 공유 블록을 반복하는 looped/recurrent-depth 구조가 같은 깊이의 비반복 모델에 근접하거나 경쟁력 있는 성능을 낼 수 있다고 보고한다. citeturn25view0turn29view0turn31view0

## 제공 자료에서 읽히는 아키텍처

당신이 제공한 첫 번째 도식은 사실상 OpenMythos 문서와 거의 같은 메시지를 담고 있다. 핵심은 **Input Embedding → Prelude → 고정 입력 표현 `e` 확보 → 공유 Recurrent Block 반복 → Coda → LM Head**라는 골격이다. 저장소 문서는 Prelude 뒤의 표현을 `e`로 고정한 뒤, 같은 recurrent block을 `n_loops`만큼 반복하고, 이 루프 안에 **loop-index 신호**, **MoE FFN**, **depth-wise LoRA**, **LTIInjection**, **ACT halting**을 배치한다고 명시한다. 또한 `e`를 매 루프에 변하지 않은 채로 다시 주입하는 것이 깊은 반복에서도 입력 신호가 떠내려가지 않게 하는 핵심 불변식이라고 설명한다. 제공된 첫 번째 도식이 강조하는 “silent recurrence + frozen input reinjection”은 이 설명과 정확히 맞물린다. citeturn18view0turn2view1turn2view2turn2view3turn2view4

두 번째 도식은 더 중요한 설계 철학을 드러낸다. 그것은 **토큰 공간 CoT**와 **잠재 공간 반복 추론**을 분리해야 한다는 점이다. 최근 연구는 looped transformer가 `T`번의 루프로 `T`단계 CoT를 잠재적으로 모사할 수 있으며, reasoning 성능이 effective depth에 따라 스케일링된다고 보고한다. 동시에 다른 연구는 깊이-반복 모델 내부에서 사람이 읽을 수 있는 형태의 latent CoT가 항상 뚜렷하게 나타나는 것은 아니고, 반복 수를 늘려도 명시적 reasoning step을 외부화한 모델보다 못할 수 있다고 지적한다. 다시 말해, **잠재 반복 추론은 강력한 계산 축**이지만, **가시적 짧은 scratchpad를 완전히 버리면 학습과 디버깅이 어려워질 수 있다**는 뜻이다. citeturn25view0turn29view0turn34view0

이 해석은 원조 계열의 선행 연구와도 연결된다. Universal Transformer는 공유 블록을 반복하여 시퀀스 표현을 계속 갱신하는 구조를 제안했고, ACT를 통해 위치별로 필요한 연산 횟수를 다르게 줄 수 있음을 보였다. 최근 recurrent-depth 연구는 이 기본 아이디어를 더 밀고 나가서, **systematic generalization**, **depth extrapolation**, 그리고 동시에 **overthinking**이라는 실패 모드를 함께 논의한다. 즉 MultimodalText는 단순히 “레이어를 아끼는 구조”가 아니라, **테스트 시점에 추론 깊이를 늘릴 수 있는 계산형 모델**로 설계하는 편이 맞다. citeturn28view0turn28view1turn31view0

## 권장 모델 구조

내가 권하는 구조는 한 문장으로 요약하면 **“작은 텍스트 추론 코어 + 동결 비전 타워 + 얕은 멀티모달 Prelude + 깊은 공유 반복 블록”**이다. 멀티모달 브리지를 가볍게 두고, 진짜 reasoning은 반복 블록이 맡게 해야 한다. 이 방향은 **frozen vision encoder + lightweight bridge**를 제안한 BLIP-2와, **임의로 interleaved된 이미지·텍스트 시퀀스**를 다룬 Flamingo, 그리고 **동적 해상도와 멀티모달 위치 부호화**를 강조한 Qwen2-VL의 공통점을 취하는 설계다. PaliGemma가 SigLIP 계열 비전 인코더와 디코더형 언어 모델을 결합한 공개 VLM으로 강한 전이 성능을 보인 점도 이 선택을 뒷받침한다. citeturn20view0turn19view1turn20view1turn19view2

권장 사양은 **두 단계**로 나누는 것이 좋다.  
첫째는 **Strict-100M 변형**이다. 이 변형은 “총 trainable core를 가급적 1억 내외로” 맞추고 싶을 때 쓴다. 숨김차원 768, `Prelude 2층 + 공유 recurrent 1블록 + Coda 2층`, GQA, 작은 top-2 MoE, 가벼운 멀티모달 bridge를 쓰면, **동결 비전 인코더를 제외한 trainable core가 대략 0.85억~0.95억**, 활성 파라미터는 **0.74억~0.76억 수준**으로 맞출 수 있다.  
둘째는 **Preferred 연구 변형**이다. 숨김차원 896, `Prelude 2층 + 공유 recurrent 1블록 + Coda 2층`, routed expert 16개와 shared expert 2개, top-2 routing, depth-wise LoRA rank 8, bridge 6M 내외로 잡으면, **trainable core는 약 1.25억~1.30억**, 활성 파라미터는 **약 1억 전후**가 된다. 내 판단으로는 이 두 번째 변형이 “앞선 대화에서 말한 100M급 감각”을 유지하면서도 수학 추론과 멀티모달 인터페이스의 균형이 가장 좋다.

이때 **반복 깊이와 저장 파라미터를 분리해서 생각**해야 한다. OpenMythos 문서도 `n_loops`를 학습 시 값보다 추론 시 더 크게 줄 수 있다고 설명하고 있으며, 최근 recurrent-depth 연구 역시 학습보다 더 깊은 추론을 테스트 시 반복으로 확보할 수 있다고 본다. 따라서 파라미터 예산은 **폭(width)과 브리지 품질**에 쓰고, 더 어려운 입력은 **추론 시 loop 수를 8→12→16으로 늘려 처리**하는 것이 옳다. citeturn18view0turn2view4turn31view0

어텐션은 **v1에서는 GQA부터 시작**하는 편을 권한다. GQA는 “KV head 수를 query head보다 줄이되 MQA보다 품질 저하를 줄이는 중간점”이고, 원 논문은 GQA가 MHA에 가까운 품질을 유지하면서 MQA에 가까운 속도를 낸다고 보고한다. 반면 MLA는 KV cache를 강하게 압축할 수 있어 긴 문맥과 대량 시각 토큰에서 유리하지만, 구현 복잡도와 디버깅 비용이 높다. OpenMythos가 GQA와 MLA를 모두 지원하도록 설계한 것도 이 교체 가능성을 의식한 것으로 보인다. 그래서 **v1은 GQA**, **v2는 문서·스크린샷 입력이 길어질 때 MLA로 교체**하는 순서가 가장 실용적이다. citeturn32view1turn32view0turn18view0

FFN은 도식과 저장소 문서대로 **Prelude/Coda는 dense SwiGLU**, **recurrent block만 sparse MoE**로 두는 것이 좋다. 이 배치는 안정적이다. 초기/말단 블록은 입력의 모달 정렬과 출력 형상을 다루고, 진짜 domain specialization은 반복 블록 안에서 일어나게 만들 수 있기 때문이다. 또한 MoE는 최근 연구가 제안한 것처럼 **fine-grained routed experts + always-on shared experts** 형태가 좋다. 내 권장값은 Preferred 변형에서 `n_experts=16, n_shared=2, top_k=2`, Strict-100M에서는 `n_experts=8~12, n_shared=1, top_k=2`다. 이 구성은 expert specialization을 늘리면서도 routing 붕괴를 덜 유발한다. citeturn33view0turn18view0

토크나이저는 일반 LLM 관성대로 큰 BPE를 쓰기보다, **32k SentencePiece 또는 byte-BPE를 쓰되 숫자와 기호를 절대 분해하지 않는 규칙**을 넣는 편이 낫다. 더 중요한 것은 숫자 내부 위치를 따로 알려주는 **number-span positional embedding**이다. 최근 산술 연구는 transformer가 자릿수 위치를 추적하지 못하는 것이 핵심 병목이라고 보고했고, 숫자의 시작점 기준 상대 위치를 주는 embedding이 addition, sorting, multiplication까지 크게 밀어 올린다고 보였다. 길이 일반화 연구도 scratchpad와 position coupling이 operand 길이와 개수 모두에 대한 일반화를 크게 높인다고 보고한다. Mathematics Dataset가 짧고 기호 밀도가 높은 형식이라는 점까지 감안하면, MultimodalText는 “더 긴 context”보다 “더 정확한 숫자 위치 정보”에 투자해야 한다. citeturn26view0turn27view0turn12view0

비전 입력은 **동결 비전 타워 + 가벼운 resampler/bridge**로 시작하되, **고정 32개 또는 64개 visual token으로 지나치게 압축하지 않는 것**이 중요하다. 문서, 스크린샷, 수식 노트처럼 텍스트가 많은 이미지는 과도한 압축에서 바로 성능이 무너진다. Qwen2-VL과 Open-Qwen2VL이 공통으로 강조하는 것이 바로 **dynamic resolution**과 **multimodal sequence packing**이다. 따라서 MultimodalText는 이미지당 visual token 수를 **96~384 범위에서 가변적**으로 두고, 작은 그림은 적게, 텍스트가 많은 캡처·슬라이드는 많이 할당해야 한다. 입력 포맷은 Flamingo처럼 `<image><text><image><text>` 식 interleaving을 허용하고, joint Prelude에서 한두 번 cross-modal 융합한 뒤 얻은 representation을 `e`로 고정하여 반복 블록 안으로 넣는 방식이 가장 효율적이다. citeturn19view1turn20view1turn35view0turn18view0

## 데이터와 CoT 자동화

데이터는 **공식 생성기 + 직접 만든 합성기 + 렌더링 엔진**의 세 층으로 가는 것이 맞다. Mathematics Dataset의 장점은 이미 생성 코드가 공개돼 있고, TensorFlow Datasets 빌더가 모듈 이름과 split 구조를 정리해 두었다는 점이다. 따라서 텍스트 단계에서는 굳이 정적 말뭉치를 오래 붙잡지 말고, **훈련 시점에 계속 새 문제를 뽑아내는 streaming generation**을 쓰는 편이 좋다. 이렇게 해야 rote memorization보다 규칙 습득을 유도하기 쉽다. citeturn1view2turn10search0turn12view0

시동용 텍스트 모듈은 다음 조합이 가장 적절하다. `arithmetic__add_or_sub`, `arithmetic__add_sub_multiple`, `comparison__pair`, `comparison__sort`, `measurement__conversion`, `numbers__base_conversion`, `numbers__place_value`, `numbers__round_number`, `algebra__linear_1d`를 **Tier A**로 두고, 이후 `arithmetic__mixed`, `numbers__div_remainder`, `numbers__gcd`, `numbers__lcm`, `algebra__sequence_next_term`, `polynomials__evaluate`를 **Tier B**로 올리는 식이다. 이 구성이 좋은 이유는 단순 산술, 비교, 자리값, 단위변환, 1차 방정식까지는 비교적 짧은 절차를 요구하고, `mixed`나 polynomial evaluation은 중간 상태를 실제로 저장·조작해야 해 모델의 약점을 빨리 드러내기 때문이다. 원 논문도 transformer가 “여러 수의 덧셈”에는 90% 이상 성능을 보였지만, 괄호가 들어가는 mixed arithmetic에서는 성능이 약 50%로 떨어졌다고 보고한다. citeturn13view3turn14view4turn14view0turn13view4

멀티모달 확장은 “기존 텍스트 문제를 그림 문제로 바꾸는 렌더링 계층”에서 시작하는 것이 제일 효율적이다. 예를 들어 `place_value`는 숫자 카드나 자리값 표로, `comparison__sort`는 숫자 막대나 number line으로, `measurement__conversion`은 단위표와 용기 그림으로, `linear_1d`는 balance-style 도식으로 렌더링할 수 있다. 이렇게 하면 **정답이 이미 보장된 텍스트 문제를 그대로 이미지 문제로 변환**할 수 있고, 텍스트-이미지 정렬 오류를 최소화할 수 있다. 그 위에 문서 캡처, 표, 강의 슬라이드, 손글씨 노트, 저해상도 스캔 같은 스타일 변형을 얹으면 “text-rich image” 학습 데이터가 무한에 가깝게 생성된다. 이 방식은 MathVista가 보여 준 것처럼 시각 수학 벤치마크가 단순 OCR이 아니라 **정교한 시각 인식 + 조합 추론**을 동시에 요구한다는 사실과도 잘 맞는다. citeturn21view0

CoT는 **완전 노출형 하나만** 쓰지 말고, **세 가지 모드로 자동 생성**하는 편이 좋다.  
하나는 `answer-only`다. 모델이 잠재 반복 추론만으로 답을 내게 한다.  
둘째는 `short-structured scratchpad`다. 예를 들어 덧셈은 자리 정렬과 carry만, 방정식은 양변 이동과 최종 정리만, base conversion은 나눗셈 나머지열만 보여준다.  
셋째는 `verifier trace`다. 이미 생성된 답이나 scratchpad가 맞는지 짧게 재검산하게 한다.  

이 세 모드를 섞어야 하는 이유는 분명하다. 작은 transformer는 **포맷과 중간 단계가 좋아질수록 sample complexity와 수렴 속도가 좋아진다**는 보고가 있고, arithmetic 일반화 연구도 task-specific scratchpad와 위치 결합이 길이 일반화를 크게 높인다고 말한다. 반면 깊이-반복 모델만으로 latent CoT가 충분히 해석 가능하고 강하게 자란다는 증거는 아직 제한적이다. 따라서 MultimodalText는 “잠재 반복 추론을 주축으로 삼되, 짧은 구조화 scratchpad를 보조적으로 학습시킨다”가 맞다. 훈련 비율은 초기에 `answer-only 50% / short scratchpad 35% / verifier 15%`, 후반부에는 `answer-only 35% / short scratchpad 45% / verifier 20%` 정도를 권한다. citeturn26view1turn27view0turn34view0turn25view0

멀티모달 pretraining은 “모든 것을 다 모으는 대형 웹 크롤링”보다 **정제된 소규모 고품질**이 낫다. Open-Qwen2VL은 2B 모델을 29M image-text pairs와 5B packed multimodal tokens로 효율적으로 학습했고, low-to-high dynamic resolution과 sequence packing의 중요성을 강조했다. 이 결과를 그대로 숫자로 복사할 필요는 없지만, **당신이 만들려는 100M급 연구 모델은 훨씬 더 소량의 데이터에서도 충분히 의미 있는 멀티모달 적응이 가능하다**는 점을 시사한다. 내 권장은 텍스트 문제를 렌더링한 합성 멀티모달 데이터를 주력으로 삼고, 일반 이미지-텍스트·문서-텍스트 데이터는 “브리지 안정화” 용도로만 제한적으로 섞는 것이다. citeturn35view0turn20view0

## 학습 절차와 파라미터 예산

훈련 절차는 **텍스트 코어를 먼저 안정화하고, 비전 브리지는 나중에 붙이는 순서**가 가장 좋다. 첫 단계는 text-only math pretraining이다. 여기서는 digit-aware tokenizer, number-span embedding, GQA, recurrent block, ACT만으로 시작한다. 둘째 단계는 visible scratchpad와 verifier trace를 섞어 reasoning format을 가르친다. 셋째 단계에서 동결된 vision encoder와 작은 resampler를 붙여, 이미 렌더링해 둔 합성 멀티모달 문제를 학습한다. BLIP-2가 제안한 frozen encoder + lightweight Querying Transformer 접근은 이 단계에 특히 잘 맞는다. 처음부터 비전 타워까지 함께 end-to-end로 흔들면, 작은 코어에서는 텍스트 추론이 자리 잡기 전에 시각 잡음을 배우느라 시간을 허비할 가능성이 크다. citeturn20view0turn18view0

loop 스케줄은 **고정값 하나보다 범위 샘플링**이 낫다. 최근 recurrent-depth 연구는 depth extrapolation이 단순히 테스트 시 loop를 늘린다고 자동으로 생기는 것이 아니라, **훈련 시 사용한 recurrence 전략의 영향을 강하게 받는다**고 말한다. 그래서 나는 초반에는 `2~6 loops`, 중반에는 `4~8 loops`, 후반에는 `6~8 loops`처럼 범위를 샘플링하고, 추론 시에는 `8 / 12 / 16 loops` 세 모드를 노출하는 구성을 권한다. 이렇게 하면 쉬운 예시는 ACT가 일찍 멈추고, 어려운 예시는 더 깊게 생각하게 만들 수 있다. 동시에 `n_loops`를 더 늘릴수록 무조건 좋아지는 것이 아니라 overthinking이 생길 수 있으므로, 각 태스크별 최적 loop 곡선을 반드시 따로 측정해야 한다. citeturn31view0turn28view1turn18view0turn2view2

안정성 측면에서는 제공된 도식과 OpenMythos 설명을 그대로 따르는 편이 낫다. 즉 반복 업데이트는 단순 residual accumulation이 아니라 **LTI-style stable injection**으로 구성하고, `A·h + B·e + transform(h)` 꼴을 쓰되 `rho(A) < 1`이 유지되도록 감시해야 한다. 여기에 ACT의 ponder cost, MoE load-balancing loss, answer span 가중치, scratchpad token mask를 더하면 된다. 이 조합이 필요한 이유는 반복 모델의 장점이 “아무 생각 없이 많이 도는 것”이 아니라, **입력을 잃지 않고 필요한 위치에만 계산을 더 쓰는 것**이기 때문이다. citeturn2view3turn18view0turn28view1

파라미터 예산은 **총 저장 파라미터와 활성 파라미터를 분리해서 관리**해야 한다. Preferred 변형은 대략 **1억 활성 / 1.25억~1.30억 trainable core**인데, 이때 recurrent block 내부의 MoE 때문에 “저장된 expert 전체”와 “토큰당 실제 사용 expert”가 다르다. 이 사고방식은 recent MoE 설계와도 일관된다. 특히 entity["company","DeepSeek","ai company"] 계열 연구는 sparse expert routing과 KV cache 압축을 통해 stored capacity와 active compute를 분리하는 것이 비용-성능 측면에서 중요하다고 보여 준다. MultimodalText도 같은 철학을 취하면 된다. 작은 연구 모델에서는 거대한 총 파라미터보다 **적당한 active width + 반복 깊이 + 정확한 데이터 형식**이 더 중요하다. citeturn32view0turn33view0turn25view0

만약 이미 작은 dense decoder checkpoint가 있다면, 또 다른 좋은 경로는 **recursive conversion**이다. Relaxed Recursive Transformers는 standard pretrained transformer를 recursive model로 바꾸고 depth-wise LoRA를 넣으면, 같은 급의 vanilla model보다 낫고 원래 더 큰 dense model의 성능 상당 부분을 회복할 수 있다고 보여 준다. 따라서 “처음부터 전부 scratch”만이 답은 아니다. 네가 이미 작은 언어 모델을 가지고 있다면, **그 체크포인트를 반복 구조로 변환한 뒤 math curriculum으로 계속 학습**하는 길도 충분히 실용적이다. citeturn25view1

## 평가 프로토콜

텍스트 평가는 반드시 **모듈별 interpolation / extrapolation / longer-than-train**의 세 층으로 나눠야 한다. 원 벤치마크는 원래부터 interpolation test와 extrapolation test를 분리해 설계됐고, builder 역시 모듈별로 이 split를 제공한다. 따라서 “평균 accuracy 하나”만 보면 안 된다. 최소한 `add_or_sub`, `add_sub_multiple`, `mixed`, `base_conversion`, `place_value`, `linear_1d`, `measurement__conversion`에 대해 **정답 정확도**, **loop budget별 성능 곡선**, **visible scratchpad on/off**, **number-span embedding on/off**를 따로 기록해야 한다. 특히 `mixed`와 composed 계열에서 loop 수 증가가 실제 성능 증가로 이어지는지가 핵심 지표다. citeturn10search0turn12view0turn13view4turn18view0

멀티모달 평가는 세 단계가 좋다.  
첫째는 **자체 렌더링한 DM-Math-Visual 세트**다. 여기서 텍스트 문제와 시각 문제의 정답이 정확히 같아야 하므로, 모델이 “시각 인식 때문에” 틀렸는지, “추론 때문에” 틀렸는지 쉽게 분해할 수 있다.  
둘째는 **MathVista**다. 이 벤치마크는 6,141개 예시를 포함하고, 강한 모델에게도 여전히 어렵다. 작은 연구 모델에게 SOTA를 기대하면 안 되지만, 이 세트를 통해 차트·도식·문서형 수학 문제에서 실제 전이 여부를 볼 수 있다.  
셋째는 **UniGeo / MathCheck-GEO** 류다. UniGeo는 계산 문제 4,998개와 증명 문제 9,543개를 포함해 geometry-style sequence generation을 볼 수 있고, MathCheck-GEO는 단순 정답률이 아니라 **robust mathematical reasoning**을 더 엄격하게 본다. 요약하면, MultimodalText는 본질적으로 “MathVista SOTA 경쟁”보다 “같은 크기의 비반복 baseline보다 확실히 나은 시각-수학 generalization”을 성공 기준으로 잡는 것이 현실적이다. citeturn21view0turn22view0turn21view2

성공 기준은 이렇게 두는 것이 좋다. 텍스트 easy 모듈에서는 interpolation이 빨리 높아져야 하고, `mixed`나 composed 계열에서 **loop 수 증가가 실제 정답 향상**으로 이어져야 하며, 렌더링된 멀티모달 세트에서는 text-only 성능 대비 손실이 작아야 한다. 반대로 실패 신호는 명확하다. ACT가 거의 항상 최대 loop까지 가는 경우, 값은 잘 맞는데 scratchpad가 붕괴하는 경우, text-only에서는 맞던 문제를 rendered screenshot에서는 못 맞는 경우, longer extrapolation에서 성능이 오른 뒤 다시 내려가는 overthinking 곡선이 나타나는 경우다. 최근 recurrent-depth 연구가 지적한 overthinking을 고려하면, “깊게 돌렸더니 더 나빠지는 지점”을 반드시 찾아야 한다. citeturn31view0turn34view0

## 최종 권고

내 최종 권고는 단순하다. **처음부터 거대한 멀티모달 RDT를 만들지 말고, 작은 텍스트 RDT를 먼저 완성하라.** 그 뒤에 **동결 비전 타워와 가변 해상도 브리지**를 붙여라. 이 순서는 공개 연구와 제공된 도식이 모두 가리키는 가장 안전한 길이다. OpenMythos가 보여 주는 `Prelude → frozen e → shared recurrent block → Coda` 구조는 그대로 유지하되, 멀티모달 요소는 Prelude 앞단과 bridge 쪽에 몰아 넣는 편이 가장 합리적이다. 그래야 시각 입력이 많아져도 “반복 블록이 매번 픽셀을 다시 보느라” 계산을 낭비하지 않는다. citeturn18view0turn20view0turn20view1

또 하나의 핵심은 **latent reasoning과 visible CoT를 적으로 보지 말라**는 점이다. 잠재 반복 추론은 이 모델의 주축이어야 한다. 하지만 작은 모델에서는 instructive formatting과 short scratchpad가 여전히 매우 큰 도움을 주며, latent CoT가 자동으로 충분히 강력하고 해석 가능한 형태로 자란다는 보장은 없다. 그래서 MultimodalText는 기본 모드에서는 “조용히 생각하고 짧게 답하는 모델”이어야 하지만, 훈련과 디버깅에서는 `explain` 제어 토큰으로 **짧은 구조화 reasoning을 외부화할 수 있는 이중 모드**를 유지하는 것이 맞다. citeturn26view1turn25view0turn34view0

정리하면, 내가 선택할 설계는 이것이다. **v1은 GQA 기반의 100M-active-class recurrent-depth text core**, **숫자 위치를 따로 주는 tokenizer/embedding**, **MoE는 recurrent block에만 적용**, **ACT와 LTI-style input injection으로 안정화**, **공식 math generator로 무한 합성**, **짧은 scratchpad 자동 생성**, **동결 vision encoder + dynamic-resolution bridge로 멀티모달 확장**이다. 이 구성이 지금까지의 공개 근거를 가장 일관되게 활용하면서도, 당신이 원한 “작은 규모에서 실제로 만들 수 있는 수학-멀티모달 연구 모델”에 가장 가깝다. citeturn18view0turn25view0turn31view0turn35view0turn21view0