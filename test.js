var fs = require('fs')
if(fs.existsSync('./public/005930,1.png')){ // 파일이 존재한다면 true 그렇지 않은 경우 false 반환
    console.log('발견')
}
else{
    console.log('없음')
}