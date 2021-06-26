// Initialize button with user's preferred color
let selectSusik = document.getElementById("select-susik");



// When the button is clicked, inject setPageBackgroundColor into current page
selectSusik.addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    window.close();
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      function: setSusikBox,
    });
  });
  
  
  function setSusikBox() {
    document.body.style.cursor = "cell";
    document.querySelector('#susik-box').style.backgroundColor = 'none';
    document.querySelector('#susik-box').setAttribute("data-activate", "true");
    document.querySelector('#overlay').style.display = 'block';
    document.querySelector('#show-box').style.display = 'block';
    
  }