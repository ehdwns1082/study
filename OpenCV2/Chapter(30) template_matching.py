# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220576634778&categoryNo=66&parentCategoryNo=0&viewDate=&currentPage=3&postListTopCurrentPage=1&from=postView&userTopListOpen=true&userTopListCount=10&userTopListManageOpen=false&userTopListCurrentPage=3
'''
템플릿 매칭이란 어떤 이미지에서 부분 이미지를 검색하고 찾는 방법이다.
여기서 말한 부분 이미지를 템플릿 이미지라고 한다.

1. 템플릿 이미지의 중심을 (x,y) 라고 두고, 템플릿 이미지를 타켓 이미지 위에 둔다.
2. 템플릿 이미지로 덮힌 타겟 이미지 부분의 픽셀값과 템플릿 이미지의 픽셀값을 특정 수학 연산으로 비교한다.
3. 2의 값을 R(x,y) 라고 하면, 타겟 이미지 전체를 훑으며 비교한 결과인 R(x,y) 전체는 타겟 이미지 보다 작은 이미지가 된다.
   만약 타겟 이미지의 사이즈가 (W,H) 이고, 템플릿 이미지의 사이즈가 (w,h) 라면, 결과 이미지의 크기는 (W-w+1, H-h+1) 크기이다.

   
'''

import cv2
import numpy as np

