var express = require('express')
  , http = require('http')
  , path = require('path');
var bodyParser = require('body-parser')
  , static = require('serve-static');
var app = express();
app.set('port', process.env.PORT || 3500);

app.use(bodyParser.urlencoded({ extended: false }))

app.use(bodyParser.json())

app.use(static(path.join(__dirname, 'public')));
app.use(function(req, res, next) {
	console.log('요청받음');

	var paramcode = req.body.code || req.query.code;
	var paramtime = req.body.time || req.query.time;
	
	res.writeHead('200', {'Content-Type':'text/html;charset=utf8'});
	res.write('<h1>요청 수락됨</h1>');
	res.write('<div><p>요청한 주식 : ' + paramcode + '</p></div>');
	res.write('<div><p>학습 횟수 : ' + paramtime + '</p></div>');
    res.write('<div><p>결과 페이지 : localhost:3500/'+paramcode+','+paramtime+'.png</p></div>');
	res.end();
    console.log(paramcode, paramtime);
    const fs = require('fs');
    fs.writeFile('require.txt', paramcode+'\n'+paramtime, (err) => {
        if (err) console.log('Error: ', err);
        else console.log('File created');
});
    fs.writeFile('check.i', ' ', (err) => {
    if (err) console.log('Error: ', err);
    else console.log('check');
});
});


// Express 서버 시작
http.createServer(app).listen(3500, function(){
  console.log('start server');
});