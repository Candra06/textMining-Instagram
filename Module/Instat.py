# Selenium Libraries for Web Scraping

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.keys import Keys
import os
import json
import time
import csv



class Instat():
    
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument('start-maximized')
        options.add_argument("--disable-blink-features=AutomationControlled")
        # options.add_extension("ig-exporter.crx");
        options.add_argument("load-extension=./IG-Exporter")


        download_dir = os.path.join(os.getcwd(), "Data")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)

        self.driver = webdriver.Chrome(options=options)
        self.once = True
        self.mainuser = ""
    
    def loadPage(self, url):
        self.driver.get(url) # Go to the url
        self.hold(8) # Wait for the Page to completely load
        
    def hold(self, seconds):
        time.sleep(seconds)
        
    def saveCookie(self):
        cookie = self.driver.get_cookies()
        with open("./Module/cookie.json", "w") as jsonfile:
            json.dump(cookie, jsonfile)
            
    def deleteCookie(self):
        self.driver.delete_all_cookies()
        
    def tabAction(self, times):
        for i in range(times):
            ActionChains(self.driver)\
                .key_down(Keys.TAB)\
                .perform()
                
    def enterAction(self):
        ActionChains(self.driver)\
            .key_down(Keys.ENTER)\
            .perform()
            
    def tabShiftAction(self, times):
        for i in range(times):
            ActionChains(self.driver)\
                .key_down(Keys.SHIFT)\
                .key_down(Keys.TAB)\
                .perform()
            ActionChains(self.driver)\
                .key_up(Keys.SHIFT)\
                .perform()
    
    def userNameLink(self, link):
        return link.split("/")[-2]
    
    def mainUserProfile(self):
            self.tabAction(9)
            self.enterAction()
            self.hold(5)
            url = self.driver.current_url
            print(url)
            self.mainuser = self.userNameLink(url)
            
    def getFollowers(self, users = None, exhaustlimit = 500):
        if users == None:
            users = [self.mainuser]
        for user in users:
            self.loadPage(f"https://www.instagram.com/{user}") # Didnt access directly to /followers because you wouldnt get the count of followers
            self.tabAction(11)    
            try:
                active = self.driver.switch_to.active_element
                active = active.find_element(By.TAG_NAME, "span")
                active = active.find_element(By.TAG_NAME, "img")
            except:
                active = self.driver.switch_to.active_element
            if active.tag_name == "img":
                self.tabAction(6)
            else:
                self.tabAction(5)
            active = self.driver.switch_to.active_element
            no_followers = active.find_element(By.TAG_NAME, "span").text
            if no_followers[-1] == "M":
                no_followers = int(float(no_followers[:-1]) * 1000000)
            elif no_followers[-1] == "K":
                no_followers = int(float(no_followers[:-1]) * 1000)
            else:
                no_followers = int(no_followers.replace(",", ""))
            
            print(f"User : {user} has {no_followers} followers")
            if no_followers < exhaustlimit:
                exhaustlimit = no_followers
            self.enterAction()
            self.hold(2)
            self.tabAction(1)
            close_button = self.driver.switch_to.active_element
            if user == self.mainuser:
                self.tabAction(1)
                active = self.driver.switch_to.active_element
                if active.tag_name == "input":
                    self.tabAction(1)
            else:
                self.tabAction(1)
                active = self.driver.switch_to.active_element
                if active.tag_name == "input":
                    self.tabAction(1)
                active = self.driver.switch_to.active_element
                try:
                    a = active.find_element(By.TAG_NAME, "img")
                except:
                    a = None
                if a == None:
                    active = self.driver.switch_to.active_element
                    if active.text == self.mainuser:
                        self.tabAction(1)
                else:
                    self.tabAction(1)
                    active = self.driver.switch_to.active_element
                    if active.text == self.mainuser:
                        self.tabAction(1)
                    
            
            # Define Some Variables
            count = 0
            follow = None
            blue = None
            tabs = None
            
            with open(f"./Data/Followers/{user}_followers.csv", mode = "a+", newline = "") as file:
                writer = csv.writer(file)
                writer.writerow(["Profile Name", "Profile Link", "Profile Image Link", "Following", "Blue Check"])
                
                while(count < exhaustlimit):
                    active = self.driver.switch_to.active_element
                    # print("this is a ", active)
                    temp = 0
                    while((active == close_button)):
                        if temp == 5:
                            break;
                        self.tabShiftAction(1)
                        self.hold(2)
                        self.tabAction(1)
                        active = self.driver.switch_to.active_element
                        temp += 1
                        # print(active.text)
                    if temp == 5:
                        break;
                    active = self.driver.switch_to.active_element
                    # print("this is b ", active)

                    try:
                        imgtag = active.find_element(By.TAG_NAME, "img")
                    except:
                        imgtag = None
                    profile_link = active.get_attribute("href")
                    print(active.get_attribute("href"))
                    print(profile_link)
                    profile_name = self.userNameLink(profile_link)
                    
                    if imgtag == None:
                        profile_img = "False"
                        try:
                            blue_check = self.driver.switch_to.active_element.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "svg")
                            if blue_check.tag_name == "svg":
                                blue = "True"
                        except:
                            blue = "False"
                        self.tabAction(1)
                        active = self.driver.switch_to.active_element
                        if active.text == "Follow":
                            if user == self.mainuser:
                                tabs = 2
                            else:
                                tabs = 1
                            follow = "False"
                        elif active.text == "Following":
                            follow = "True"
                            tabs = 1
                        else:
                            follow = "True"
                            tabs = 1
                    else:
                        profile_img = imgtag.get_attribute("src")
                        self.tabAction(1)
                        try:
                            blue_check = self.driver.switch_to.active_element.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "svg")
                            if blue_check.tag_name == "svg":
                                blue = "True"
                        except:
                            blue = "False"
                        self.tabAction(1)
                        active = self.driver.switch_to.active_element
                        if active.text == "Follow":
                            if user == self.mainuser:
                                tabs = 2
                            else:
                                tabs = 1
                            follow = "False"
                        elif active.text == "Following":
                            follow = "True"
                            tabs = 1
                        else:
                            follow = "True"
                            tabs = 1
                    count += 1
                        
                    writer.writerow([profile_name, profile_link, profile_img, follow, blue])
                    print(f"Profile Name : {profile_name}\nProfile Link : {profile_link}\nProfile Image : {profile_img}\nFollow : {follow}\nBlue Check : {blue}")
                    self.tabAction(tabs)
            close_button.click()
        
    def getFollowing(self, users = None, exhaustlimit = 500):
        if users == None:
            users = [self.mainuser]
        for user in users:
            self.loadPage(f"https://www.instagram.com/{user}") # Didnt access directly to /followers because you wouldnt get the count of followers
            self.tabAction(11)    
            try:
                active = self.driver.switch_to.active_element
                active = active.find_element(By.TAG_NAME, "span")
                active = active.find_element(By.TAG_NAME, "img")
            except:
                active = self.driver.switch_to.active_element
            if active.tag_name == "img":
                print("reached")
                self.tabAction(7)
            else:
                self.tabAction(6)
            active = self.driver.switch_to.active_element
            no_followers = active.find_element(By.TAG_NAME, "span").text
            if no_followers[-1] == "M":
                no_followers = int(float(no_followers[:-1]) * 1000000)
            elif no_followers[-1] == "K":
                no_followers = int(float(no_followers[:-1]) * 1000)
            else:
                no_followers = int(no_followers.replace(",", ""))
            print(f"User : {user} has {no_followers} following")
            if no_followers < exhaustlimit:
                exhaustlimit = no_followers
            self.enterAction()
            self.hold(2)
            self.tabAction(1)
            close_button = self.driver.switch_to.active_element
            if user == self.mainuser:
                self.tabAction(1)
                active = self.driver.switch_to.active_element
                if active.tag_name == "input":
                    self.tabAction(1)
            else:
                self.tabAction(1)
                active = self.driver.switch_to.active_element
                if active.tag_name == "input":
                    self.tabAction(1)
                active = self.driver.switch_to.active_element
                try:
                    a = active.find_element(By.TAG_NAME, "img")
                except:
                    a = None
                if a == None:
                    active = self.driver.switch_to.active_element
                    if active.text == self.mainuser:
                        self.tabAction(1)
                else:
                    self.tabAction(1)
                    active = self.driver.switch_to.active_element
                    if active.text == self.mainuser:
                        self.tabAction(1)
            
            # Define Some Variables
            count = 0
            follow = None
            blue = None
            tabs = None
            
            with open(f"./Data/Following/{user}_following.csv", mode = "a+", newline = "") as file:
                writer = csv.writer(file)
                writer.writerow(["Profile Name", "Profile Link", "Profile Image Link", "Following", "Blue Check"])
                
                while(count < exhaustlimit):
                    active = self.driver.switch_to.active_element
                    # print("this is a ", active)
                    temp = 0
                    while((active == close_button)):
                        if temp == 5:
                            break;
                        self.tabShiftAction(1)
                        self.hold(2)
                        self.tabAction(1)
                        active = self.driver.switch_to.active_element
                        temp += 1
                        # print(active.text)
                    if temp == 5:
                        break;
                    active = self.driver.switch_to.active_element
                    # print("this is b ", active)

                    try:
                        imgtag = active.find_element(By.TAG_NAME, "img")
                    except:
                        imgtag = None
                    profile_link = active.get_attribute("href")
                    # print(active.get_attribute("href"))
                    # print(profile_link)
                    profile_name = self.userNameLink(profile_link)
                    
                    if imgtag == None:
                        profile_img = "False"
                        try:
                            blue_check = self.driver.switch_to.active_element.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "svg")
                            if blue_check.tag_name == "svg":
                                blue = "True"
                        except:
                            blue = "False"
                        self.tabAction(1)
                        active = self.driver.switch_to.active_element
                        if active.text == "Follow":
                            if user == self.mainuser:
                                tabs = 2
                            else:
                                tabs = 1
                            follow = "False"
                        elif active.text == "Following":
                            follow = "True"
                            tabs = 1
                        else:
                            follow = "True"
                            tabs = 1
                    else:
                        profile_img = imgtag.get_attribute("src")
                        self.tabAction(1)
                        try:
                            blue_check = self.driver.switch_to.active_element.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "div")
                            blue_check = blue_check.find_element(By.TAG_NAME, "svg")
                            if blue_check.tag_name == "svg":
                                blue = "True"
                        except:
                            blue = "False"
                        self.tabAction(1)
                        active = self.driver.switch_to.active_element
                        if active.text == "Follow":
                            if user == self.mainuser:
                                tabs = 2
                            else:
                                tabs = 1
                            follow = "False"
                        elif active.text == "Following":
                            follow = "True"
                            tabs = 1
                        else:
                            follow = "True"
                            tabs = 1
                    count += 1
                        
                    writer.writerow([profile_name, profile_link, profile_img, follow, blue])
                    # print(f"Profile Name : {profile_name}\nProfile Link : {profile_link}\nProfile Image : {profile_img}\nFollow : {follow}\nBlue Check : {blue}")
                    self.tabAction(tabs)
            close_button.click()

    def login(self):
        if os.path.getsize("./Module/cookie.json") == 0: # If cookie isn't stored
            with open("./Module/config.json", "r") as jsonfile:
                config = json.load(jsonfile)
                username = config["username"]
                password = config["password"]
                self.driver.find_element(By.NAME, "username").send_keys(username)
                self.driver.find_element(By.NAME, "password").send_keys(password)
            self.driver.find_element(By.CSS_SELECTOR, "._acan._acap._acas._aj1-").click()
            self.hold(7) # Wait for 7 seconds to load the page properly
            self.saveCookie() # Save the cookie for future use
            
        else:
            self.deleteCookie()
            with open("./Module/cookie.json", "r") as jsonfile:
                cookie = json.load(jsonfile)
                for c in cookie:
                    self.driver.add_cookie(c)
            self.driver.refresh()
            self.hold(7)
    def scrapByLink(self, url):
        self.loadPage("https://chromewebstore.google.com/detail/ig-exporter-scraper-expor/nmgmcehdhckaehgfokcomaboclhbdpkb?hl=en-US&utm_source=ext_sidebar");
        self.driver.find_element(By.CSS_SELECTOR, ".UywwFc-vQzf8d").click() # Click on the IG Exporter Scraper Exporter
        while True:
            # check if text change to "Remove from Chrome"
            try:
                self.driver.find_element(By.CSS_SELECTOR, ".UywwFc-vQzf8d").text == "Remove from Chrome"
                print("Extension installed successfully")
                break
            except:
                print("Extension not installed yet, waiting for 2 seconds")
                time.sleep(2)
        self.loadPage("https://www.instagram.com/accounts/login/")
        self.login() # Login into Instagram
        self.hold(2) # Wait for the page to load
        self.loadPage("chrome-extension://nmgmcehdhckaehgfokcomaboclhbdpkb/options.html") 
        self.driver.find_element(By.NAME, "user").send_keys(url)
        self.driver.find_element(By.CSS_SELECTOR, "#pane-options .row .option-button:nth-child(1)").click()

        # check text with id workStatus
        finish = False
        while True:
            try:
                workStatus = self.driver.find_element(By.ID, "workStatus")
                if workStatus.text == "Finished":
                    self.driver.find_element(By.CSS_SELECTOR, ".export-button-left").click()
                    print("Scraping finished")
                    finish = True
                    break
            except:
                pass
        
        time.sleep(2)
        
        # 
        
        if finish:
            self.driver.quit()
            print("Browser closed")
        else:
            print("Scraping not finished")
        

    def getComments(self, url):
        comment_element = self.driver.find_element(By.CSS_SELECTOR, ".x78zum5.xdt5ytf.x1iyjqo2 > div")
        print("DIV COUNT:"+len(comment_element))
    def getPostId(self, url):
        # Get the post ID from the URL
        if "instagram.com/p/" in url:
            post_id = url.split("/")[-2]
            return post_id
        else:
            print("Invalid URL format. Please provide a valid Instagram post URL.")
            return None
    def getLatestFile(self):
        files = os.listdir("./Data")
        
        # read csv only
        csv_files = [f for f in files if f.endswith(".csv")]
        # sort by date
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join("./Data", x)), reverse=True)
        # show file with path
        file = os.path.join("./Data", csv_files[0])
        print("Latest file: " + file)
        # read csv file
        with open(file, "r") as csvfile:
            reader = csv.reader(csvfile)
            # read first line
            header = next(reader)
            # read all lines
            for row in reader:
                data = row[0].split(";")
                pirnt()
                # row to json
                # show text field value in  first row
                # print(row[0])
                # print("LOOKING FOR TEXT")
    def textCleaning(self, text):
        # remove all special characters
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\t", " ")
        text = text.replace("  ", " ")
        # clean all to make it readable
        text = text.replace("  ", " ")
        text = text.replace(" ", " ")
        # remove all special characters
        text = text.replace("!", "")
        text = text.replace("@", "")
        text = text.replace("#", "")
        text = text.replace("$", "")
        text = text.replace("%", "")
        text = text.replace("^", "")
        text = text.replace("&", "")
        text = text.replace("*", "")
        
        return text
    
        