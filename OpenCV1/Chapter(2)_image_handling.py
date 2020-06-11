import cv2

imgolor = cv2.imread('image_ball.jpg', cv2.IMREADOLOR)
# retval(=리턴밸류) = cv.imread(file_name[,flags(=옵션)])
# example of flag)
# cv2.IMREADOLOR = 투명도 정보를 가진 알파채널을 무시하고 다 컬러로 읽음
# cv2.IMREAD_GRAYSCALE = 그레이스케일 즉 흑백으로 읽음
# cv2.IMREAD_UNCHANGED = 투명도 정보를 가진 알파채널을 포함해서 이미지를 컬러로 읽음

cv2.namedWindow('Show Image')
# named 함수를 호출하여 window에 컬러이미지가 보이도록 함
# namedWindsow는 윈도우 창에 track bar를 붙일 떄 빼고는 생략 가능함
# 1st argument = 윈도우 식별자(window title)

cv2.imshow('Show Image', img_CHolor)
# 1st argument = 윈도우 식별자(window title), 2nd argumnet = 윈도우에 보여질 이미지에 대한 변수

cv2.waitKey(0)
# waitkey 함수는 argument로 지정한 시간만큼 사용자의 키보드 입력을 대기, unit은 second
# waitkey(0) = 키보드 입력 무한대기, 키보드 입력이 있기 전까지 윈도우를 화면에 계속 띄워놓음
# 아무키나 누르면 다음 단계로 넘어감. 다음 단계가 없을 시 창 닫음

img_gray = cv2.cvtColor(img_CHolor, cv2.COLOR_RGB2GRAY)
# cvtColor 함수로 컬러 이미지를 흑백 이미지로 변환하기
# 1st argument =  변환할 대상 이미지, 2st argument = 변환할 색 공간(RGB to GRAY)
# cvtColor 함수의 Return이 그레이 스케일 이미지가 됨

cv2.imshow("Show Image", img_gray)
# imshow 함수의 윈도우 이름을 새로운 이름으로 지정하면 컬러 이미지와 그레이 이미지 윈도우를 동시에 띄울 수 있다.

cv2.waitKey(0)

cv2.imwrite('saved_image.jpg', img_gray)
# imwrite 함수로 변환된 이미지를 파일로 저장 할 수 있음

cv2.destroyAllWindows()
# 프로그램 종료 전 윈도우에 대한 자원을 해제시킴








