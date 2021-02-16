# Camera_Recognition
Raspberry Pi B+ Camera test (VS in Windows connect B+ project)

- Opencv use C++


정사각형 검출 및 가로 및 세로 검출, 넓이, 꼭지점 위치 계산
  1. 영상으로부터 이미지 획득
  
  2. to Gray 로 변경
  
  3. Smoothing (blur)
    https://m.blog.naver.com/PostView.nhn?blogId=ledzefflin&logNo=220503016163&proxyReferer=https%3A%2F%2Fwww.google.com%2F
    cv.bilateralFilter() is highly effective in noise removal while keeping edges sharp.
    
  4. Thresholding
    어떤 주어진 임계값(threshold)보다 밝은 픽셀들은 모두 흰색으로, 그렇지 않은 픽셀들은 모두 검은색으로 바꾸는 것
      경계 구분을 위함
      
  5. Canny Edge Detect
    input-image	: 8-bit input image.
    모서리(선분) 찾기
    
  6. Contours
    윤곽 : 동일한( 색상 강도를 가진 부분의 가장 자리 경계를 연결한 선
    cv::findContours
    cv::drawContours
    
  7. Serach Rectangles of Polygons
    다각형 근사
    approxPolyDP()
    contourArea()
