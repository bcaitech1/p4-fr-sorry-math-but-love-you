chrome.runtime.onMessage.addListener(function(msg, sender, sendResponse) {
    //스크린 캡쳐를 하는 코드
    chrome.tabs.captureVisibleTab(null, {
        format : "png",
        quality : 100
    }, function(data) {
        sendResponse(data);
    });
    return true;
});

