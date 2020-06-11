import cv2

cap = cv2.VideoCapture(0) # 웹캠 불러오기
# VideoCapture 객체 생성, 카메라가 2대라면 argument가 0인 객체와 argument가 1인 객체 생성
# VideoCapture() 함수에 파일명을 넣으면 저장된 비디오를 불러오고 0.1 등을 넣으면 입력 디바이스 순서에 따라 실시간 촬영 frame을 불러옴


while(True) :
    ret,img_color = cap.read()
    # 캡쳐 객체의 read() 함수를 호출하여 cap 으로부터 한 프레임을 가져온다
    # 프레임을 제대로 읽으면 ret(리턴)값이 True 실패하면 False가 나타남
    # 파이썬에서 자주 쓰는 표현은 아니지만 알고 있으면 유용함. 그냥 외우셈

    if ret == False :
        continue

    cv2.imshow("Color", img_color)
    #캡쳐된 이미지를 화면에 보여준다.

    if cv2.waitKey(1) & 0xFF == 27 :
        break
    # ESC 누르면 창 종료
    # ESC 누르면 ret 값이 27이 나온다.
    ''' 
    1. cv2.waitKey()는 32비트 정수 값을 반환한다.(플랫폼에 따라 다를 수 있음).
    2. 키보드 입력은 ASCII 이므로 8비트 정수 값이다. 
    3. 0xFF(16진수)는 2진수로 11111111인 상수이다. -> 8비트
    4. cv2.waitKey() 와 0xFF 를 AND(&) 연산하면 cv2.waitKey()값이 무엇이든 간에 입력의 마지막 8비트만 남는다.
    5. 비트 수가 맞춰졌으므로 키보드 입력값에 맞는 상수를 'cv2.waitKey() & 0xFF == '  우측에 넣어 주면 끝~
    '''
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
    # https://cnpnote.tistory.com/entry/PYTHON-cv2waitKey-1%EC%9D%98-0xFF%EB%8A%94-%EB%AC%B4%EC%97%87%EC%9E%85%EB%8B%88%EA%B9%8C

cap.release()
# 마지막으로 오픈한 cap 객체를 cap.release() 함수를 이용해 반드시 해제한다. 그리고 생성한 모든 윈도우를 제거한다.

cv2.destroyWindow("Color")
# 윈도우 창 종료

# https://zzsza.github.io/data/2018/01/23/opencv-1/
