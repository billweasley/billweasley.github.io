(function(){var t,i=function(t,i){return function(){return t.apply(i,arguments)}};t=function(){function t(t){this._eventToAccept=i(this._eventToAccept,this),void 0===window.console&&(window.console=function(){}),this.options=$.extend({},this.defaults,t),$.cookie("accepted-cookies")?this._printAnalytics():""!==document.referrer&&document.referrer.indexOf(document.domain)?this._ifSafePath()||($.cookie("accepted-cookies","true",{expires:712}),this._printAnalytics()):(this._initialize(),this.options.onInit())}return t.prototype.defaults={ga:{id:"UA-XXXXXXXX-X",domain:"plasm.it"},position:"bottom",safe_paths:[],selector:"body",template:'<div class="privacy-cookies"> <div class="privacy-cookies-contents"> <div class="privacy-contents-left"> <p>Questo sito utilizza cookie tecnici e analitici per migliorare l’esperienza dell’utente e raccogliere informazioni statistiche sul suo comportamento online.<br>Proseguendo la navigazione, l’utente presta automaticamente il proprio consenso al loro utilizzo. Per maggiori informazioni si rimanda alla <a data-safe-event="safe-event" href="{link}" target="_blank">cookie policy</a>.</p> </div> <div class="privacy-contents-right"> <button type="button">{btn_text}</button> </div> </div> </div>',btn_text:"Accetta!",link:"#",opt_cookie:{expires:712,path:"/",domain:""},onInit:function(){},onAccepted:function(){},onClose:function(){}},t.prototype._initialize=function(){var t,i;return i=this,t=this._t(this.options.template,{btn_text:this.options.btn_text,link:this.options.link}),$("body").prepend(t),$("div.privacy-cookies").hide().delay(200).slideDown(400),this._setPadding(),$("div.privacy-cookies").find("button").on("click",function(t){return function(){return t._accept()}}(this)),$(window).bind("click.privacy, scroll.privacy",this._eventToAccept),$(window).on("resize",function(t){return function(){return t._setPadding()}}(this))},t.prototype._eventToAccept=function(t){return $(t.target).data("safe-event")?void 0:this._accept()},t.prototype._accept=function(){return $.cookie("accepted-cookies","true",this.options.opt_cookie),$(window).unbind("click.privacy, scroll.privacy",this._eventToAccept),this._printAnalytics(),this._removePadding(),this.options.onAccepted(),$("div.privacy-cookies").slideToggle(400,function(t){return function(){return $("div.privacy-cookies").remove(),t.options.onClose()}}(this))},t.prototype._ifSafePath=function(){var t;return null!=(t=0===$.inArray(String(window.location.pathname),this.options.safe_paths))?t:{"false":!0}},t.prototype._setPadding=function(){var t;return $(this.options.selector).css((t={},t["padding-"+this.options.position]=$("div.privacy-cookies").height(),t)),$("div.privacy-cookies").css(""+this.options.position,0)},t.prototype._removePadding=function(){var t;return $(this.options.selector).css((t={},t["padding-"+this.options.position]="",t))},t.prototype._printAnalytics=function(){return ga("create",this.options.ga.id,this.options.ga.domain),ga("send","pageview")},t.prototype._t=function(t,i){var o;for(o in i)t=t.replace(new RegExp("{"+o+"}","g"),i[o]);return t},t}(),window.Privacy=t}).call(this);