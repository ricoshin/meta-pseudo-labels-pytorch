- [x] 스케쥴러
- [x] 테스트 코드
- [x] 학습 진행률 출력
- [x] tqdm
- [x] Performance monitor
- [x] 베스트 성능 값도 같이 저장
- [x] 테스트 코드 acc 안올라가는지 확인
- [x] 시그널 디버거 도입
- [x] cosine annealing 확인 ( + 텐서보드에 쓰기 )
- [x] 텐서보드
- [x] soft labeling
- [x] .yaml autocompletion 가능하도록 상대경로 입력
- [x] config 내용 stdio에 출력
- [x] 최종 결과 뿌리는 파일 스트림 생성
- [x] eval_only 구현
- [x] 학습 이어갈때는 last load, 테스트할때는 best load (or 선택)
- [x] valid도 tqdm 출력
- [x] eval_only의 경우 config도 함께 load
- [x] colorize warning msgs
- [x] 현재 어떤 태그로 훈련 중인지 콘솔에 지속적으로 출력
- [x] MPL: student학습시는 teacher gradient 일시적 차단(그래프는 유지)

- [x] UDA_labeled: TSA(loss)
- [x] UDA_unlabeled: confidence_threshold(target) / softmax_temp(target)

- [ ] gradient clipping
- [ ] cfg.uda.on 제거
- [ ] .yaml 내용 정리
- [ ] supervised, label smoothing, randaugment: use student hyper params

- [ ] TFrecord 경로 수정 (지금 수정하면 이어서 학습 불가 - 연기)
- [ ] default = simple + cutout
- [ ] 실행 자동화 코드 구현
- [ ] 하이퍼 파라미터 튜닝 코드 구현 ( ray? cfg 외부변경 가능하도록 )
- [ ] test standard deviation

---

- [x] Supervised
- [x] Label Smoothing
- [x] RandAugment
- [x] UDA
- [x] MPL
