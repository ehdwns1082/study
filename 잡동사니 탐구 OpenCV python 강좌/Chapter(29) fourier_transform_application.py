# https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220568857153&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=false&from=postList

'''
푸리에 변환은 이미지를 주파수 영역으로 전환하여 이미지 프로세싱 작업을 수행할 수 있게 해는 도구이고,
주파수 영역에서 작업이 끝나면 역푸리에 변환(Inversion Fourier Transform:IFT)을 수행하여 원래 이미지 영역으로 되돌려서 이미지 프로세싱 결과를 확인 할 수 있다.

LPF(Low Pass Filter) 는 낮은 주파수 대력만 통과시키는 필터이고, HPF(High Pass Filter) 는 높은 주파수 대역만 통과시키는 필터이다.
이미지에서 LPF 를 사용하면 낮은 주파수 대역만 남아있는 이미지가 되므로 blur 효과를 가진 이미지가 된다.
이미지에서 HPF 를 사용하면 높은 주파수 대역만 남아있는 이미지가 되므로 사물의 edge 나 노이즈 등만 남아 있는 이미지가 된다.

푸리에 변환을 통해 주파수 영역으로 옮긴 이미지로 주파수 작업을 수행하면 보다 다양한 필터링 작업을 수행할 수 있다.
'''

