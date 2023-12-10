## 사용법
1. server.js, ai.py 실행
2. localhost/main.html
3. 페이지 지시 이행

## 통신체계(코드전반해석)
- 사용자

1>.사용자접근 2>.값입력 14>>.그래프 호출 16<. 그래프 확인
- html

3>.html을 통해 사용자와 통신
- node.js(server) 0.서버구동

4>.로컬파일에 사용자가 입력한 값 require.txt파일 생성 5>. check.i 파일 생성 15<. 그래프 출력
- 로컬파일

6<. check.i파일 확인 7<. check.i파일 삭제 8>. require.txt 파일 읽기 10<. 이미 연산된 파일인지 확인 13<. 그래프 저장
- 파이썬 0.시스템구동 9. require.txt 파일 내용 확인 11. 딥러닝 모델 생성 12. 그래프 생성


## Known Issue
- 파일 경로에 한국어가 포함되면 실행이 원활하지 않음
- 주식 코드를 불러오지 못하면 ai.py가 중단됨
- 실행 직후 또는 첫 요청에서 ai.py가 간헐적으로 중단됨
- 연산중에 요청이 2번 이상 발생하면 마지막 요청에만 반응함

## Future
각각의 계층이 어떤것을 주고 받는지에 관한 정보를 담은 그림을 추후 업데이트 예정
