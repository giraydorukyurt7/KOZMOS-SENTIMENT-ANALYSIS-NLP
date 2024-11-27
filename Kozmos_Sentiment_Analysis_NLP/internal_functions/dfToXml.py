import pandas as pd

def dfToXml(df,filename, filedirectory,xsl_href,index_=False, encode="utf-8"):
    fullfilename = filedirectory+"/"+filename
    df.to_xml(fullfilename,index=index_,root_name="data") #save df as xml
    #add xsl referance
    with open(fullfilename,"r",encoding=encode) as file:
        xml_content = file.read()
    xslt_reference = '<?xml-stylesheet type="text/xsl" href="%s"?>\n' % xsl_href
    xml_with_xslt  = xml_content.replace("<?xml version='1.0' encoding='utf-8'?>",
                                         "<?xml version='1.0' encoding='utf-8'?>\n" + xslt_reference)
    with open(fullfilename,"w",encoding=encode) as file:
        file.write(xml_with_xslt)