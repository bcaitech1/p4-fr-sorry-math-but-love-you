const SERVERL_URL='http://118.67.134.149:6012/susik_recognize';

fetch(chrome.runtime.getURL('/template.html')).then(r => r.text()).then(html => {
    document.body.insertAdjacentHTML('beforeend', html);
  });

let selectBoxX = -1
let selectBoxY = -1


document.body.addEventListener('click', e=>{
    //하단에 박스 닫는 버튼 클릭시
    if (e.target.id =='susik-close'){
        let showBox = document.querySelector('#show-box');
        let susikBox = document.querySelector('#susik-box');
        let susikOutput = document.querySelector('#susik-output');
        let susikImage = document.querySelector('#susik-image');
        let susikLatex = document.querySelector('#susik-output-latex');

        //관련된 UI들 초기화
        showBox.style.display = 'none';
        susikBox.style.width='0px';
        susikBox.style.height='0px';
        susikBox.style.top="-1px";
        susikBox.style.left="-1px";
        susikOutput.value = '';
        susikImage.src = '';
        susikLatex.src = '';        
    }
    //fix 버튼 클릭시
    if (e.target.id == 'fix-text'){
        //latex image를 변경된 text에 맞추어 갱신
        document.querySelector('#susik-output-latex').src = "http://latex.codecogs.com/gif.latex?" + document.querySelector('#susik-output').value
    }
    //copy 버튼 클릭시
    if (e.target.id == 'copy-text'){
        //text copy
        copyText()
    }
});


document.body.addEventListener('mousedown', e => {
    var isActivated = document.querySelector('#susik-box').getAttribute("data-activate");
    let susikBox = document.querySelector('#susik-box');
    susikBox.style.display='block';

    if(isActivated=="true"){
        let x = e.clientX;
        let y = e.clientY;
        //Select Box의 시작점을 현재 마우스 클릭 지점으로 등록
        selectBoxX = x;
        selectBoxY = y;
        
        //Susik Box이 위치와 사이즈를 현재 지점에서 초기화
        susikBox.style.top = y+'px';
        susikBox.style.left = x+'px';
        susikBox.style.width='0px';
        susikBox.style.height='0px';
    }
});


//캡쳐가 준비된 상태에서 (마우스 클릭이 된 상태) 드래그시 박스 사이즈 업데이트
document.body.addEventListener('mousemove', e => {
    try{
        var susikBox = document.querySelector('#susik-box');
        var isActivated = susikBox.getAttribute("data-activate");
    }catch(e){
        return;
    }
    
    //팝업에서 Start 버튼을 클릭하고, select 박스의 값이 초기값이 아닌 상태인 경우 시작
    if(isActivated=="true" && (selectBoxX != -1 && selectBoxY != -1)){
        let x = e.clientX;
        let y = e.clientY;

        //Select 박스(susik-box)의 가로 세로를 마우스 이동에 맞게 변경
        width = x-selectBoxX;
        height = y-selectBoxY;
        
        susikBox.style.width = width+'px';
        susikBox.style.height = height+'px';
        
    }
});

// 마우스 드래그가 끝난 시점 (드랍)
document.body.addEventListener('mouseup', e => {
    let susikBox = document.querySelector('#susik-box');
    let isActivated = susikBox.getAttribute("data-activate");
    
    //만약 팝업의 start 버튼을 클릭한 후의, 그냥 취소
    if(isActivated=="false"){
        return ;
    }

    // 다음 이벤트가 ??
    susikBox.setAttribute("data-activate", "false");
    
    
    
    let x = parseInt(selectBoxX);
    let y = parseInt(selectBoxY);
    let w = parseInt(susikBox.style.width);
    let h = parseInt(susikBox.style.height);
    
    //캡쳐 과정이 끝났으므로, susik-box 관련된 내용 초기화
    selectBoxX = -1;
    selectBoxY = -1;
    
    
    susikBox.style.display='none';
    susikBox.style.width='0px';
    susikBox.style.height='0px';
    susikBox.style.top="-1px";
    susikBox.style.left="-1px";

    //Overaly 화면 안보이게 초기화
    document.querySelector('#overlay').style.display='none';
    //마우스 Cursor도 원래 커서로 초기화
    document.body.style.cursor = "default";


    //200ms 정도의 시간차를 두고 서버로 현재 캡쳐된 이미지를 전송
    //시간차를 안두면, 박스와 오버레이 화면이 같이 넘어갈 수 있음
    setTimeout(function(){
        chrome.runtime.sendMessage({text:"hello"}, function(response) {
            var img=new Image();
            img.crossOrigin='anonymous';
            img.onload=start;
            img.src=response;
            
            function start(){
                //화면 비율에 따라 원래 설정한 좌표 및 길이와 캡쳐본에서의 좌표와 길이가 다를 수가 있어서, 그에 대응하는 비율을 곱해줌
                ratio = img.width/window.innerWidth;
                
                
                var croppedURL=cropPlusExport(img,x*ratio,y*ratio,w*ratio,h*ratio);
                var cropImg=new Image();
                cropImg.src=croppedURL;
                document.querySelector('#susik-image').src = croppedURL;
                fetch(SERVERL_URL, {
                    method: 'POST',
                    body: JSON.stringify({"image":croppedURL}), // data can be `string` or {object}!
                    headers:{
                        'Content-Type': 'application/json'
                    }
                }).then(res => res.json())
                .then(response => {    
                    document.querySelector('#susik-output').value = response['result'];
                    document.querySelector('#susik-output-latex').src = "http://latex.codecogs.com/gif.latex?" + response['result'];
                });
            }
    
        });
    },200);
   
    
});

//전체 스크린샷을 crop하는 함수
function cropPlusExport(img,cropX,cropY,cropWidth,cropHeight){
    
    
    var canvas1=document.createElement('canvas');
    var ctx1=canvas1.getContext('2d');
    canvas1.width=cropWidth;
    canvas1.height=cropHeight;
    
    ctx1.drawImage(img,cropX,cropY,cropWidth,cropHeight,0,0,cropWidth,cropHeight);
    
    return(canvas1.toDataURL());
  }

//textbox의 내용을 copy하는 함수
function copyText() {
    var obj = document.getElementById("susik-output");
    obj.select(); //인풋 컨트롤의 내용 전체 선택
    document.execCommand("copy"); //복사
    obj.setSelectionRange(0, 0); //선택영역 초기화
  }
